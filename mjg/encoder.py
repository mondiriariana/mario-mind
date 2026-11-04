"""Frozen V-JEPA video encoders: V-JEPA 2 (Hugging Face) and V-JEPA 2.1 (Meta's own code).

A decision step sees a short clip of ``clip_len`` screens (stride ``clip_stride`` raw
frames) ending just before the step. The clip is resized to ``size`` x ``size`` and encoded
by the frozen backbone. The token grid [T/2, n, n, D] is averaged over time and pooled to a
``grid x grid`` spatial map, giving ``z_t`` of shape [grid*grid, D]. The backbone is never
trained: eval mode, requires_grad=False, torch.no_grad.

``FrozenVJEPA2(spec)`` picks the backbone from ``spec``:
  a Hugging Face id or folder   V-JEPA 2, e.g. facebook/vjepa2-vitl-fpc64-256 (ViT-L, 256 px)
  vjepa2.1:<folder>             V-JEPA 2.1, e.g. vjepa2.1:${MJG_VJEPA21_G} (ViT-G, 2B, 384 px); the
                                folder holds Meta's checkpoint (vjepa2_1_*.pt from
                                dl.fbaipublicfiles.com) and a clone of facebookresearch/vjepa2 in
                                code/; V-JEPA 2.1 is not in transformers
"""
from __future__ import annotations

import glob
import os
import sys

import torch
import torch.nn.functional as F

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


class _Frozen(torch.nn.Module):
    """Shared preprocessing and pooling; subclasses set model, dim, patch, tubelet and _tokens."""

    def _freeze(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def preprocess(self, clips_u8: torch.Tensor) -> torch.Tensor:
        """uint8 [B, T, H, W, 3] (any device) -> normalized [B, T, 3, size, size]."""
        x = clips_u8.to(self.device, non_blocking=True)
        B, T, H, W, _ = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(B * T, 3, H, W).float().div_(255)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False,
                          antialias=True)
        x = x.view(B, T, 3, self.size, self.size)
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        return x.to(self.dtype)

    def _pool(self, h: torch.Tensor, B: int, T: int, grid: int) -> torch.Tensor:
        """tokens [B, T/tubelet * n * n, D] -> time-averaged, grid-pooled float16 [B, grid*grid, D]."""
        n = self.size // self.patch
        h = h.view(B, T // self.tubelet, n, n, self.dim).float().mean(1)  # [B, n, n, D]
        h = h.permute(0, 3, 1, 2)
        h = F.adaptive_avg_pool2d(h, grid)  # [B, D, g, g]
        return h.flatten(2).transpose(1, 2).to(torch.float16)

    @torch.no_grad()
    def _pooled(self, x: torch.Tensor, grid: int) -> torch.Tensor:
        """normalized [B, T, 3, size, size] -> float16 [B, grid*grid, D]."""
        B, T = x.shape[:2]
        return self._pool(self._tokens(x), B, T, grid)

    @torch.no_grad()
    def forward(self, clips_u8: torch.Tensor) -> torch.Tensor:
        """uint8 clips [B, T, H, W, 3] -> pooled latents float16 [B, grid*grid, D]."""
        return self._pooled(self.preprocess(clips_u8), self.grid)

    @torch.no_grad()
    def encode_crops(self, crops01: torch.Tensor) -> torch.Tensor:
        """Float clips in [0, 1], [B, T, 3, h, w] (any crop size) -> one latent per clip,
        the mean over all of the backbone's tokens: float16 [B, D]."""
        B, T = crops01.shape[:2]
        x = crops01.to(self.device).flatten(0, 1).float()
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = x.view(B, T, 3, self.size, self.size)
        x = ((x - MEAN.to(x.device)) / STD.to(x.device)).to(self.dtype)
        return self._pooled(x, 1)[:, 0]


class HFVJEPA2(_Frozen):
    """V-JEPA 2 through Hugging Face transformers (hub id or local folder)."""

    def __init__(self, model_id: str | None = MODEL_ID, grid: int = 4, size: int = 256,
                 dtype=torch.bfloat16, device="cuda", attn="sdpa"):
        super().__init__()
        from transformers import VJEPA2Model
        model_id = os.path.expandvars(model_id or MODEL_ID)  # hub id or local folder, e.g. ${MJG_VJEPA_G}
        self.model_id = model_id
        self.model = VJEPA2Model.from_pretrained(model_id, dtype=dtype, attn_implementation=attn).to(device)
        self._freeze()
        self.grid, self.size, self.dtype, self.device = grid, size, dtype, device
        self.dim = self.model.config.hidden_size
        self.patch = self.model.config.patch_size
        self.tubelet = self.model.config.tubelet_size

    def _tokens(self, x):
        return self.model(pixel_values_videos=x, skip_predictor=True).last_hidden_state


# Meta's hub builders for the V-JEPA 2.1 checkpoints (src/hub/backbones.py in facebookresearch/vjepa2)
_V21_BUILDERS = {"vitb": "vjepa2_1_vit_base_384", "vitl": "vjepa2_1_vit_large_384",
                 "vitg": "vjepa2_1_vit_giant_384", "vitG": "vjepa2_1_vit_gigantic_384"}


class MetaVJEPA21(_Frozen):
    """V-JEPA 2.1 with Meta's code: the builder from src/hub/backbones.py, our local checkpoint.

    Loads the ``target_encoder`` weights (what Meta's own loader uses), with the ``module.`` and
    ``backbone.`` prefixes removed as Meta's loader does, strictly. The encoder's default output is
    the final layer after its LayerNorm, tokens [B, T/2 * n * n, D]; input is [B, C, T, H, W].
    """

    def __init__(self, folder: str, grid: int = 4, size: int = 384, dtype=torch.bfloat16, device="cuda",
                 attn=None):
        super().__init__()
        folder = os.path.expandvars(folder)
        ckpts_all = sorted(glob.glob(os.path.join(folder, "vjepa2_1_*.pt")))
        ckpts = []
        for c in ckpts_all:
            if not c.endswith(".target_encoder_bf16.pt"):  # skip the cached slim copy
                ckpts.append(c)
        if len(ckpts) != 1 or not os.path.isdir(os.path.join(folder, "code", "src")):
            raise FileNotFoundError(f"{folder}: expected one vjepa2_1_*.pt and Meta's code in code/")
        size_key = os.path.basename(ckpts[0]).split("_")[2]          # vitG / vitg / vitl / vitb
        sys.path.insert(0, os.path.join(folder, "code"))
        from src.hub import backbones
        encoder, predictor = getattr(backbones, _V21_BUILDERS[size_key])(pretrained=False)
        del predictor
        # the full checkpoint holds encoder, target encoder and predictor in fp32; keep a one-time
        # copy of just the target encoder in bf16 next to it, so parallel jobs load ~4 GB, not ~30 GB
        slim = ckpts[0][:-3] + ".target_encoder_bf16.pt"
        if os.path.exists(slim):
            state = torch.load(slim, map_location="cpu")
        else:
            raw_state = torch.load(ckpts[0], map_location="cpu", weights_only=False)["target_encoder"]
            state = {}
            for k, v in raw_state.items():
                key = k.replace("module.", "").replace("backbone.", "")
                state[key] = v.to(torch.bfloat16) if v.is_floating_point() else v
            try:
                torch.save(state, slim + f".tmp{os.getpid()}")
                os.replace(slim + f".tmp{os.getpid()}", slim)
            except OSError:
                pass
        encoder.load_state_dict(state, strict=True)
        del state
        self.model_id = f"vjepa2.1:{folder}"
        self.model = encoder.to(device=device, dtype=dtype)
        self._freeze()
        self.grid, self.size, self.dtype, self.device = grid, size, dtype, device
        self.dim, self.patch, self.tubelet = encoder.embed_dim, 16, 2

    def _tokens(self, x):
        # autocast as in Meta's code: its RoPE runs in fp32 and SDPA needs q, k, v in one dtype
        with torch.autocast("cuda", dtype=self.dtype):
            return self.model(x.permute(0, 2, 1, 3, 4))  # [B, T, C, H, W] -> Meta's [B, C, T, H, W]

    @property
    def depths(self) -> list[int]:
        """The 4 blocks Meta's model normalises and exposes (hierarchical_layers; the last = final output)."""
        return list(self.model.hierarchical_layers)

    @torch.no_grad()
    def forward_depths(self, clips_u8: torch.Tensor, grid: int | None = None) -> torch.Tensor:
        """uint8 clips [B, T, H, W, 3] -> float16 [B, len(depths), grid*grid, D]: each of the 4 exposed
        blocks after its own LayerNorm, averaged over time and pooled like forward()."""
        x = self.preprocess(clips_u8)
        B, T = x.shape[:2]
        self.model.out_layers = self.depths
        try:
            with torch.autocast("cuda", dtype=self.dtype):  # see _tokens
                outs = self.model(x.permute(0, 2, 1, 3, 4))
        finally:
            self.model.out_layers = None
        pooled = []
        for h in outs:
            pooled.append(self._pool(h, B, T, grid or self.grid))
        return torch.stack(pooled, 1)

    @torch.no_grad()
    def forward_depths_last(self, clips_u8: torch.Tensor, grid: int | None = None):
        """Like forward_depths(), plus the final block's most recent time step only (the clip's last
        tubelet = its last 2 frames), pooled the same way. Returns (float16 [B, len(depths), grid*grid, D],
        float16 [B, grid*grid, D])."""
        x = self.preprocess(clips_u8)
        B, T = x.shape[:2]
        g = grid or self.grid
        self.model.out_layers = self.depths
        try:
            with torch.autocast("cuda", dtype=self.dtype):  # see _tokens
                outs = self.model(x.permute(0, 2, 1, 3, 4))
        finally:
            self.model.out_layers = None
        n = self.size // self.patch
        last = outs[-1].view(B, T // self.tubelet, n, n, self.dim)[:, -1].float().permute(0, 3, 1, 2)
        last = F.adaptive_avg_pool2d(last, g).flatten(2).transpose(1, 2).to(torch.float16)
        pooled = []
        for h in outs:
            pooled.append(self._pool(h, B, T, g))
        return torch.stack(pooled, 1), last


def FrozenVJEPA2(model_id: str | None = MODEL_ID, **kw):
    """The frozen encoder for ``model_id``: ``vjepa2.1:<folder>`` -> V-JEPA 2.1, else V-JEPA 2 (HF)."""
    spec = os.path.expandvars(model_id or MODEL_ID)
    if spec.startswith("vjepa2.1:"):
        kw.setdefault("size", 384)
        return MetaVJEPA21(spec[len("vjepa2.1:"):], **kw)
    return HFVJEPA2(spec, **kw)
