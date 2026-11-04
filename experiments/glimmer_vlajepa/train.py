"""VLA-JEPA (arXiv 2602.10098) on Muse Glimmer 30B, for Mario: the world-model fine-tune that precedes SFT and RL.

Glimmer sees the current screenshot + a neutral prompt (no action, no future), followed by K learnable
latent-action tokens. Their final hidden states, squeezed to K x 64, condition a small world model that
predicts the future V-JEPA 2.1 G state (newest tubelet, block 47, 2x2) at +0.53 / +1.07 / +2.0 s from the
current V-JEPA state, as in experiments/vjepa_spark_combo/latent_wm.py. Loss: masked L2 in V-JEPA space
(paper eq. 5). Future frames are only targets, never inputs.

  --mode frozen   Glimmer frozen; only the latent tokens + world model learn (control)
  --mode lora     + LoRA r16 on every linear layer of the language model (and --vision-lora: the vision tower)
  --mode full     every Glimmer weight trains (FSDP2 over the language model + vision tower; torchrun only;
                  fp32 master weights; the model kept at the end is the last step, unless --patience: then the run
                  stops after N validations without a new best and the best model is the one exported).
                  --lm-data adds a next-token loss on Glimmer's own texts so the latent objective does not erase
                  its language.
                  The trained Glimmer is saved in HF format to <out>/<tag>/glimmer (from_pretrained-able; the latent
                  tokens + world model stay in trainable.pt). Sharded checkpoints (model + optimizer + loop state)
                  go to --ckpt-dir/<tag> every --ckpt-every-min, and the run checkpoints and exits by itself when
                  the Slurm job has < --stop-margin-min left; a new job with the same arguments + --resume continues.

Data: the frame set of experiments/vjepa_spark_combo (build_set.py; plus tokens_future.npz, the future V-JEPA
tubelets) in $MJG_SCRATCH/vjepa_spark_combo/v1. Split: fold 0 of its session folds is the test set, an inner session
split of the rest gives the validation set, as in latent_wm.py. Model: $MJG_MODELS/Muse-Glimmer-30B.
Outputs in $MJG_ROOT/results/glimmer_vlajepa/<tag>/: glimmer/ (full), trainable.pt, result.json (test R^2, probes).

Single process (model split over the visible GPUs, e.g. 2x L40S):  python train.py --mode lora
Data parallel (one full copy per GPU, needs >= 80 GB GPUs):        torchrun --nproc-per-node 8 train.py ...
Full fine-tune (settings in long_run.sh):                          sbatch experiments/glimmer_vlajepa/job_long.sbatch
"""
import argparse, json, math, os, shutil, subprocess, sys, time
from datetime import timedelta
import numpy as np
import torch, torch.nn as nn, torch.distributed as dist
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

ROOT = os.environ.get("MJG_ROOT", ".")
SCRATCH = os.environ.get("MJG_SCRATCH", os.path.join(ROOT, "scratch"))
MODELS = os.environ.get("MJG_MODELS", os.path.join(ROOT, "models"))
CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(CODE, "experiments", "vjepa_spark_combo"))
from analyze import folds, load, probes  # noqa: E402
from latent_wm import r2_vw  # noqa: E402

MODEL = f"{MODELS}/Muse-Glimmer-30B"
SET = f"{SCRATCH}/vjepa_spark_combo/v1"
OUT = f"{ROOT}/results/glimmer_vlajepa"
PROMPT = "This image is a screenshot of a person playing Super Mario Bros. (NES)."
DZ = 64


class LoRA(nn.Module):
    def __init__(self, base, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base, self.scale = base, alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.base(x)
        return y + ((self.drop(x).to(self.A.dtype) @ self.A.t() @ self.B.t()) * self.scale).to(y.dtype)


def add_lora(root, r):
    n = 0
    for name, m in list(root.named_modules()):
        for cname, c in list(m.named_children()):
            if isinstance(c, nn.Linear) and not isinstance(m, LoRA):
                setattr(m, cname, LoRA(c, r, 2 * r))
                n += 1
    return n


def find_layers(model, n_layers):
    """The decoder ModuleList (len == n_layers) of the language model."""
    for name, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) == n_layers and "vision" not in name:
            return name, m
    raise RuntimeError("decoder layers not found")


def local(x):
    """The local shard of an FSDP2 DTensor (shares its storage); x itself otherwise."""
    return x.to_local() if hasattr(x, "to_local") else x


class AdamWState32(torch.optim.Optimizer):
    """AdamW whose moment estimates (m, v) are fp32 whatever the weight dtype; the update is computed in fp32 and
    rounded once into the weight. Meta's documented Glimmer full fine-tune (FSDP2, bf16 weights and gradients, fp32
    Adam state, adamw_torch_fused, lr ~1e-5; dev.meta.ai/docs/muse-glimmer/post-training). torch.optim.AdamW keeps its
    state in the weight dtype, i.e. bf16 here, too coarse for the second moment: the full fine-tune diverges.
    Works on FSDP2 DTensor shards (element-wise ops; state tensors share the weight's sharding)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):  # closure: torch.distributed.checkpoint calls step(closure=None) to create state
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["step"] = torch.zeros((), dtype=torch.float32)
                    st["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                st["step"] += 1
                t = float(st["step"])
                pl, gr, m, v = local(p), local(p.grad).float(), local(st["exp_avg"]), local(st["exp_avg_sq"])
                m.mul_(b1).add_(gr, alpha=1 - b1)
                v.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                upd = (m / (1 - b1 ** t)) / (v / (1 - b2 ** t)).sqrt().add_(g["eps"])
                w = pl.float()
                if g["weight_decay"]:
                    w.mul_(1 - g["lr"] * g["weight_decay"])
                w.add_(upd, alpha=-g["lr"])
                if pl.dtype == torch.bfloat16:
                    # stochastic rounding: an lr-1e-5 update is below bf16 resolution for most weights (|w| > ~2e-3)
                    # and round-to-nearest would drop it; rounding up or down at random keeps it on average
                    i = w.view(torch.int32)
                    i.add_(torch.randint_like(i, 0, 1 << 16)).bitwise_and_(-65536)
                pl.copy_(w)


class WMHead(nn.Module):
    """World model: s_t (+ latent z) -> future V-JEPA tubelet at each horizon, residual on the current one."""
    def __init__(self, d_s, d_h, k, n_h, d_out):
        super().__init__()
        self.n_h, self.d_out = n_h, d_out
        self.es = nn.Sequential(nn.LayerNorm(d_s), nn.Linear(d_s, 512), nn.GELU(), nn.Dropout(0.3))
        self.ez = nn.Sequential(nn.LayerNorm(d_h), nn.Linear(d_h, DZ))
        self.mlp = nn.Sequential(nn.Linear(512 + k * DZ, 1024), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(1024, n_h * d_out))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s, h, cur):
        z = self.ez(h.float()).flatten(1)  # [B, K*DZ]
        y = self.mlp(torch.cat([self.es(s), z], 1)).view(-1, self.n_h, self.d_out)
        return y + cur[:, None], z


def job_end():
    """Unix time at which this Slurm job is killed (None outside Slurm)."""
    jid = os.environ.get("SLURM_JOB_ID")
    if not jid:
        return None
    try:
        out = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%e"], capture_output=True, text=True, timeout=60).stdout
        return time.mktime(time.strptime(out.strip(), "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return None


def n_params(params):
    total = 0
    for p in params:
        total += p.numel()
    return total


def stack_rows(rows, idx):
    """np.stack of rows[i] for the indices i in idx."""
    out = []
    for i in idx:
        out.append(rows[int(i)])
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["frozen", "lora", "full"], required=True)
    ap.add_argument("--lr-full", type=float, default=1e-5, help="full fine-tune: LR of the Glimmer weights")
    ap.add_argument("--vision-lora", action="store_true")
    ap.add_argument("--k", type=int, default=8, help="latent-action tokens")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8, help="global batch per optimizer step")
    ap.add_argument("--mb", type=int, default=4, help="micro-batch per process")
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--precision", choices=["auto", "fp32-master", "bf16-sr"], default="auto",
                    help="full: fp32-master = Meta's recipe as accelerate runs it (trainable weights upcast to fp32 "
                         "master copies, FSDP2 MixedPrecisionPolicy bf16 compute + bf16 gradient reduce, fused AdamW, "
                         "fp32 state); bf16-sr = bf16 weights, fp32 Adam state, stochastic rounding (AdamWState32), "
                         "for GPUs too small for fp32 master weights; auto = fp32-master when it fits in GPU memory")
    ap.add_argument("--evals", type=int, default=6, help="validation passes during training")
    ap.add_argument("--max-train", type=int, default=0,
                    help="only the first N training frames (validation and test: the first max(8, N/4))")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--save-model", type=int, default=-1,
                    help="full: save the trained Glimmer (1/0; -1 = unless --max-train)")
    ap.add_argument("--ckpt-dir", default=f"{SCRATCH}/ckpt/glimmer_vlajepa",
                    help="full: sharded checkpoints, <dir>/<tag>")
    ap.add_argument("--ckpt-every-min", type=float, default=120, help="full: checkpoint every N minutes (0 = never)")
    ap.add_argument("--stop-margin-min", type=float, default=45,
                    help="full: checkpoint and exit when the job has < N min left")
    ap.add_argument("--pause-after-steps", type=int, default=0,
                    help="full: checkpoint and exit after N steps in this job")
    ap.add_argument("--resume", action="store_true", help="full: continue from the latest checkpoint of this tag")
    ap.add_argument("--max-skips", type=float, default=0.0, help="full: share of steps that may be skipped for a "
                    "non-finite loss / gradient before stopping (default 0: stop at the first one and report it)")
    ap.add_argument("--lm-data", default=None, help="full: also train next-token prediction on these texts (a "
                    "gen_human_reasoning.py jsonl: Glimmer's own perception + reasoning on human turns, in the agent's "
                    "conversation), so the latent objective does not erase Glimmer's language")
    ap.add_argument("--lm-weight", type=float, default=1.0, help="weight of the language loss")
    ap.add_argument("--lm-every", type=int, default=1, help="one language sample per process every N steps")
    ap.add_argument("--lm-balance", type=float, default=0.0, help="> 0: ignore --lm-weight and set the language weight "
                    "every step so the language gradient on Glimmer has this ratio to the VLA-JEPA one (1 = equal "
                    "pull; at a fixed weight the far larger language gradient would decide nearly every update). Both "
                    "norms are measured exactly, one per language step in turn (that objective's backward first, "
                    "alone), and averaged (EMA 0.9)")
    ap.add_argument("--lm-val-n", type=int, default=48, help="held-out texts for the language validation loss")
    ap.add_argument("--patience", type=int, default=0, help="full: stop after N validations without a new best val "
                    "loss (0 = off); the best model is kept (sharded, model only) and is the one exported, not the "
                    "last; with --schedule wsd: start the final cooldown instead")
    ap.add_argument("--warmup-steps", type=int, default=0, help="full: warmup length (0 = 10%% of the steps)")
    ap.add_argument("--export-at", nargs="*", default=[], help="full: at the first validation after each of these "
                    "times (YYYY-MM-DDTHH:MM, local), also export the current model (HF format) to "
                    "<out>/<tag>/glimmer_step<N> with milestone.json (its step, val loss, the best so far), so SFT and "
                    "RL can start from it while training goes on")
    ap.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine", help="full: cosine (Meta's recipe, "
                    "decays over all --epochs) or wsd = warmup, constant LR, then a (1-sqrt) cooldown to 0 (Hagele et "
                    "al. 2024, arXiv 2405.18392: matches a tuned cosine without fixing the length in advance; gains "
                    "plateau at a ~20%% cooldown). The final cooldown starts when --patience fires on the constant-LR "
                    "validations, or so that it ends at --epochs; each --export-at milestone gets its own cooldown "
                    "branch: checkpoint, cool down, validate, export, then reload the checkpoint and go on at the "
                    "constant LR")
    ap.add_argument("--cooldown-frac", type=float, default=0.2,
                    help="wsd: final cooldown length, share of the steps so far")
    ap.add_argument("--milestone-cooldown-frac", type=float, default=0.1,
                    help="wsd: milestone cooldown, share of the steps so far")
    ap.add_argument("--finish-at-limit", action="store_true", help="full: near the job's time limit, end the training "
                    "and export the best model (instead of checkpointing for --resume)")
    a = ap.parse_args()
    assert not (a.lm_data or a.patience or a.finish_at_limit) or a.mode == "full", \
        "--lm-data / --patience / --finish-at-limit are for --mode full"

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if ddp:  # long timeout: rank 0 alone writes the 60 GB model while the others wait
        dist.init_process_group("nccl", timeout=timedelta(minutes=90))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    main_proc = rank == 0
    say = (lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)) if main_proc else (lambda *x: None)
    tag = a.tag or f"{a.mode}{'_vis' if a.vision_lora else ''}_k{a.k}"
    od = f"{OUT}/{tag}"
    if main_proc:
        os.makedirs(od, exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # ---------------------------------------------------------------- data: test = session fold 0, inner val split
    lab, classes, v, _ = load(SET, need_spark=False)
    n = len(lab)
    fu = np.load(f"{SET}/tokens_future.npz")
    horizons = fu["horizons"].tolist()
    cur = fu["cur_last"].astype(np.float32).reshape(n, -1)
    fut = fu["fut_last"].astype(np.float32).reshape(n, len(horizons), -1)
    valid = fu["valid"]
    S0 = np.hstack([v["depth_mean"][:, 3], v["b47_grid2"].reshape(n, -1), cur])
    groups = (lab["sub"] + "_" + lab["ses"]).values
    tr0, te = folds(lab)[0]
    itr, iva = next(GroupShuffleSplit(1, test_size=0.15, random_state=0).split(tr0, groups=groups[tr0]))
    tr, va = tr0[itr], tr0[iva]
    if a.max_train:
        n_eval = max(8, a.max_train // 4)
        tr, va, te, tr0 = tr[:a.max_train], va[:n_eval], te[:n_eval], tr0[:a.max_train]
    mu_s, sd_s = S0[tr].mean(0), S0[tr].std(0) + 1e-6
    mu_t, sd_t = cur[tr].mean(0), cur[tr].std(0) + 1e-6
    S = torch.tensor((S0 - mu_s) / sd_s)
    C = torch.tensor((cur - mu_t) / sd_t)
    Y = torch.tensor((fut - mu_t) / sd_t)
    Mk = torch.tensor(valid)
    frames = np.load(f"{SET}/frames.npz")["frames"]
    say(f"frames {len(frames)}; train {len(tr)} val {len(va)} test {len(te)}; world {world}")

    # ---------------------------------------------------------------- model
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(MODEL)
    tok = proc.tokenizer
    tok.padding_side = "left"  # the K latent tokens are always the last K positions
    t0 = time.time()
    full = a.mode == "full"
    assert ddp or not full, "--mode full needs torchrun"
    assert full or not a.resume, "--resume is for --mode full"
    if full:  # load on CPU, shard below; Meta's full recipe: flash attention (PyTorch SDPA runs the same kernels), tf32
        model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        # cuDNN attention (PyTorch's default SDPA kernel on H200) gives NaN gradients in the vision tower; with it off
        # SDPA uses FlashAttention, as Meta's flash_attention_2 recipe does.
        torch.backends.cuda.enable_cudnn_sdp(False)
    elif ddp:
        model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                            device_map={"": int(os.environ["LOCAL_RANK"])})
    else:
        max_memory = {}
        for gpu in range(torch.cuda.device_count()):
            max_memory[gpu] = "38GiB" if gpu == 0 else "44GiB"
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="auto", max_memory=max_memory)
    model.requires_grad_(False)
    cfg = model.config
    H = cfg.text_config.hidden_size
    lname, layers = find_layers(model, cfg.text_config.num_hidden_layers)
    emb_std = float(model.get_input_embeddings().weight.float().std())
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if full else model.get_input_embeddings().weight.device
    say(f"loaded in {time.time() - t0:.0f}s; decoder at {lname}; hidden {H}; attention "
        f"{getattr(model.config, '_attn_implementation', '?')} / text "
        f"{getattr(model.config.text_config, '_attn_implementation', '?')}")
    n_lora = 0
    if a.mode == "lora":
        n_lora = add_lora(layers, a.r)
        if a.vision_lora:
            vis = None
            for nm, m in model.named_modules():
                if nm.endswith("vision_tower"):
                    vis = m
                    break
            n_lora += add_lora(vis, a.r)
    # also when frozen: the latent tokens enter at the input, so their gradient goes through all layers
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    full_params = []
    if full:
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh("cuda", (world,))
        shard_kw = {}
        inner = model.model  # lm_head is not used (we read hidden states) and stays on CPU, frozen
        inner.requires_grad_(True)
        # GiB per GPU: fp32 weights + gradients + 2 Adam states, + activations etc.
        need = 16 * n_params(inner.parameters()) / world / 2 ** 30 + 24
        have = torch.cuda.get_device_properties(dev).total_memory / 2 ** 30
        prec = a.precision if a.precision != "auto" else ("fp32-master" if need < have else "bf16-sr")
        say(f"precision {prec} (fp32 master needs ~{need:.0f} GiB per GPU, the GPU has {have:.0f} GiB)")
        up = prec == "fp32-master"  # as accelerate's fsdp2_prepare_model: fp32 master weights, bf16 compute
        if up:
            from torch.distributed.fsdp import MixedPrecisionPolicy
            shard_kw["mp_policy"] = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)

        def upcast(mod):  # one block at a time, right before it is sharded: every rank holds the whole model on CPU,
            if up:        # and upcasting it all first would need 4 x ~106 GiB of host memory (the job has 400 GiB)
                for p in mod.parameters():
                    if p.dtype != torch.float32 and not hasattr(p, "to_local"):
                        p.data = p.data.float()
        for nm, m in inner.named_modules():
            if isinstance(m, nn.ModuleList):
                for blk in m:
                    upcast(blk)
                    fully_shard(blk, mesh=mesh, **shard_kw)
        upcast(inner)  # what is left outside the blocks (embeddings, norms, projector)
        fully_shard(inner, mesh=mesh, **shard_kw)
        import gc
        gc.collect()  # drop the CPU copies the shards replaced (4 ranks share the job's 400 GiB of host memory)
        full_params = list(inner.parameters())
        say(f"FSDP2 over {world} GPUs; {n_params(full_params) / 1e9:.1f}B trainable")
    lora_params = []
    for nm, p in model.named_parameters():
        if nm.endswith((".A", ".B")):
            lora_params.append(p)
    say(f"LoRA modules {n_lora}, params {n_params(lora_params) / 1e6:.1f}M")

    emb = model.get_input_embeddings()
    latents = nn.Parameter(torch.randn(a.k, H, device=dev) * emb_std)
    k = a.k
    use_latents = [True]  # off for the language samples, which have no latent slots
    emb.register_forward_hook(lambda m, i, o: torch.cat([o[:, :-k], latents.to(o.dtype).expand(len(o), k, H)], 1)
                              if use_latents[0] else o)
    head = WMHead(S.shape[1], H, k, len(horizons), C.shape[1]).to(dev)
    groups_opt = [{"params": list(head.parameters()) + [latents], "lr": a.lr_head}]
    small = list(groups_opt[0]["params"])  # replicated on every rank: gradients all-reduced by hand
    if lora_params:
        groups_opt.append({"params": lora_params, "lr": a.lr_lora})
        small += lora_params
    if full_params:
        groups_opt.append({"params": full_params, "lr": a.lr_full, "weight_decay": 0.0})
    if full and prec == "bf16-sr":
        opt = AdamWState32(groups_opt, weight_decay=0.05)
    elif full:  # Meta's recipe: adamw_torch_fused
        opt = torch.optim.AdamW(groups_opt, weight_decay=0.05, fused=True)
    else:
        opt = torch.optim.AdamW(groups_opt, weight_decay=0.05)
    say(f"optimizer {type(opt).__name__}")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    def to_dev(batch):
        out = {}
        for kk, vv in batch.items():
            out[kk] = vv.to(dev) if torch.is_tensor(vv) else vv
        return out

    def batch_inputs(idx):
        imgs = []
        for i in idx:
            imgs.append(Image.fromarray(np.repeat(np.repeat(frames[i], 2, 0), 2, 1)))  # 2x upscale
        b = proc(text=[text] * len(idx), images=imgs, return_tensors="pt", padding=True)
        ids = torch.cat([b["input_ids"], torch.full((len(idx), k), pad_id, dtype=b["input_ids"].dtype)], 1)
        am = torch.cat([b["attention_mask"], torch.ones(len(idx), k, dtype=b["attention_mask"].dtype)], 1)
        b["input_ids"], b["attention_mask"] = ids, am
        if "mm_token_type_ids" in b:
            tt = b["mm_token_type_ids"]
            b["mm_token_type_ids"] = torch.cat([tt, torch.zeros(len(idx), k, dtype=tt.dtype)], 1)
        return to_dev(b)

    def latent_states(idx):
        out = model.model(**batch_inputs(idx))
        return out.last_hidden_state[:, -k:].to(dev)  # [B, K, H]

    def step_loss(idx, train):
        idx = np.asarray(idx)
        with torch.set_grad_enabled(train):
            h = latent_states(idx)
            p, z = head(S[idx].to(dev), h, C[idx].to(dev))
            m = Mk[idx].float()[..., None].to(dev)
            loss = (((p - Y[idx].to(dev)) ** 2) * m).sum() / (m.sum() * p.shape[-1]).clamp(min=1)
        return loss, p, z, h

    # ---------------------------------------------------------------- language loss (--lm-data)
    # Next-token loss on Glimmer's own texts, in the exact conversation they were written for (frames of the last 8
    # turns + current frames + prompt, player tag; gen_human_reasoning.py), loss on the text only. lm_head is the
    # original one, frozen (Glimmer's output layer is not tied to its embeddings); every other weight trains.
    lm_items = {}
    if a.lm_data:
        sys.path.insert(0, CODE)
        sys.path.insert(0, os.path.join(CODE, "experiments", "glimmer_imitation"))
        from train_sft import Turns
        rs = []
        for line in open(a.lm_data):
            rs.append(json.loads(line))
        dirs = set()
        for r in rs:
            dirs.add(os.path.dirname(os.path.dirname(r["file"])))
        dirs = sorted(dirs)
        for sp in ("train", "val"):
            ds = Turns(sp, 0, dirs)  # several folders: player tag in the prompt, as when the texts were written
            lm_items[sp] = []
            for r in rs:
                if r["split"] == sp and r["file"] in ds.meta:
                    lm_items[sp].append((ds, r["file"], r["i"], r["text"]))
        val_items = []
        for i in rng.permutation(len(lm_items["val"]))[:a.lm_val_n]:
            val_items.append(lm_items["val"][i])
        lm_items["val"] = val_items
        lm_head = model.get_output_embeddings().to(dev)  # frozen, replicated (~2.7 GB in bf16)
        say(f"language loss on {a.lm_data}: {len(lm_items['train'])} train / {len(lm_items['val'])} val texts "
            f"(of {len(rs)}), weight {a.lm_weight}, one per process every {a.lm_every} steps; {len(dirs)} subjects")

    def lm_loss(item):
        """Mean next-token loss on the text of one (dataset, play file, turn, text) item."""
        ds, f, i, txt = item
        msgs = ds.conversation(f, i, 8, "low")[0]
        imgs = []
        for m in msgs:
            if isinstance(m["content"], list):
                for c in m["content"]:
                    if c.get("type") == "image":
                        imgs.append(c["image"])
        p = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        b = proc(text=[p + txt], images=[imgs], return_tensors="pt")
        n_p = proc(text=[p], images=[imgs], return_tensors="pt")["input_ids"].shape[1]
        b = to_dev(b)
        use_latents[0] = False
        try:
            hid = model.model(**b).last_hidden_state[0]
        finally:
            use_latents[0] = True
        tgt = b["input_ids"][0, n_p:]
        # Glimmer's own logits (MuseGlimmerForConditionalGeneration.forward): lm_head, x output_multiplier, then a tanh
        # softcap T * tanh(x / T).
        tc = model.config.text_config
        logits = lm_head(hid[n_p - 1:-1].to(lm_head.weight.dtype)).float() * tc.output_multiplier
        logits = tc.final_logit_softcapping * torch.tanh(logits / tc.final_logit_softcapping)
        return nn.functional.cross_entropy(logits, tgt)

    def grad_norm_now():
        """Global L2 norm of the Glimmer gradients accumulated so far (FSDP shards summed over the processes)."""
        t = torch.zeros((), dtype=torch.float64, device=dev)
        for p in full_params:
            if p.grad is not None:
                t += local(p.grad).double().pow(2).sum().to(dev)
        if ddp:
            dist.all_reduce(t)
        return float(t.sqrt())

    def lm_evaluate():
        model.eval()
        tot = torch.zeros(2, device=dev)
        mine = lm_items["val"][rank::world] if ddp else lm_items["val"]
        n_each = math.ceil(len(lm_items["val"]) / world)
        with torch.no_grad():
            for j in range(n_each):  # the same number of forwards on every rank (FSDP collectives)
                it = mine[j] if j < len(mine) else lm_items["val"][0]
                l = lm_loss(it)
                if j < len(mine):
                    tot += torch.tensor([float(l), 1.0], device=dev)
        if ddp:
            dist.all_reduce(tot)
        model.train()
        return float(tot[0] / tot[1])

    def shard(idx):
        return idx[rank::world] if ddp else idx

    def batches(idx):
        """This rank's micro-batches; every rank gets the same number (FSDP needs matching collectives).
        An empty slot is filled with a dummy frame (is_dummy=True) whose output is discarded."""
        mine = shard(idx)
        nb = max(1, math.ceil(math.ceil(len(idx) / world) / a.mb))
        for b in np.array_split(mine, nb):
            yield (b, False) if len(b) else (np.asarray(idx[:1]), True)

    def evaluate(idx):
        model.eval()
        head.eval()
        tot = torch.zeros(2, device=dev)
        for b, dummy in batches(idx):
            l, _, _, _ = step_loss(b, False)
            if not dummy:
                tot += torch.tensor([float(l) * len(b), len(b)], device=dev)
        if ddp:
            dist.all_reduce(tot)
        model.train()
        head.train()
        return float(tot[0] / tot[1])

    # ---------------------------------------------------------------- gradient self-check (full)
    # Which attention kernels give finite gradients depends on the GPU. Before training, one batch with a loss that
    # depends on every layer; if a gradient is not finite, try the next SDPA kernel set and say which failed; stop if
    # none gives finite gradients (no masking: the per-step guard below stays).
    if full:
        sdp = torch.backends.cuda
        trials = [("FlashAttention (cuDNN attention off)", None),
                  ("math attention only", dict(cudnn=False, flash=False, mem=False))]
        ok_name = None
        for name, kernels in trials:
            if kernels is not None:
                sdp.enable_cudnn_sdp(kernels.get("cudnn", True))
                sdp.enable_flash_sdp(kernels.get("flash", True))
                sdp.enable_mem_efficient_sdp(kernels.get("mem", True))
                sdp.enable_math_sdp(True)
            try:
                model.train()
                h = latent_states(shard(np.asarray(tr[:a.mb * world])))
                h.float().pow(2).mean().backward()
                bad = torch.zeros(1, device=dev)
                where = []
                for nm, p in inner.named_parameters():
                    if p.grad is not None and not torch.isfinite(local(p.grad)).all():
                        where.append(nm)
                        if len(where) == 3:
                            break
                bad[0] = float(bool(where))
                if ddp:
                    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            except Exception as e:  # a bug in this check must not stop training: report it, keep the step guard
                say(f"gradient self-check itself failed ({type(e).__name__}: {e}); continuing with cuDNN attention off")
                ok_name = "unchecked, cuDNN attention off"
                opt.zero_grad(set_to_none=True)
                break
            opt.zero_grad(set_to_none=True)
            say(f"gradient self-check, {name}: "
                + ("all finite" if not bad.item() else f"NON-FINITE (rank {rank} first: {where})"))
            if not bad.item():
                ok_name = name
                break
        if ok_name is None:
            raise RuntimeError("non-finite gradients with every attention kernel set: stopping (see the lines above)")
        say(f"attention kernels for training: {ok_name}")

    # ---------------------------------------------------------------- train
    assert a.bs % (world * a.mb) == 0, "--bs must be a multiple of world size x --mb"
    steps = max(1, int(a.epochs * len(tr) / a.bs))
    every = max(1, steps // a.evals)
    wsd = full and a.schedule == "wsd"
    cool = {}  # wsd: the cooldown in progress (c0 = its first step, n = its length, kind = "milestone" / "final")
    if wsd:
        warm = a.warmup_steps or math.ceil(0.1 * steps)
        lam = lambda s: s / max(1, warm) if s < warm else (
            max(0.0, 1 - math.sqrt(min(1.0, (s - cool["c0"]) / cool["n"]))) if cool else 1.0)
    elif full:  # Meta's full recipe = transformers' cosine schedule, warmup_ratio 0.1, lr 0 at the first step
        warm = a.warmup_steps or math.ceil(0.1 * steps)
        lam = lambda s: s / max(1, warm) if s < warm else (
            0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    else:
        warm = max(1, int(0.03 * steps))
        lam = lambda s: (s + 1) / warm if s < warm else (
            0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)

    def snap():
        """CPU copy of the small trainable parts: latent tokens, world-model head, LoRA weights."""
        head_sd = {}
        for kk, vv in head.state_dict().items():
            head_sd[kk] = vv.cpu().clone()
        lora = []
        for q in lora_params:
            lora.append(q.detach().cpu().clone())
        return {"latents": latents.detach().cpu().clone(), "head": head_sd, "lora": lora}

    # full fine-tune: sharded checkpoints of everything that trains (Glimmer, latents, head, optimizer, loop state)
    ck_root = f"{a.ckpt_dir}/{tag}"
    if full:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
        holder = nn.Module()
        holder.inner, holder.head, holder.latents = inner, head, latents

    def save_ckpt(s, order, pos):
        d = f"{ck_root}/step{s:06d}"
        t = time.time()
        msd, osd = get_state_dict(holder, opt)
        dcp.save({"model": msd, "optim": osd}, checkpoint_id=d)
        if main_proc:
            torch.save(dict(step=s, order=order, pos=pos, rng=rng.bit_generator.state, sched=sched.state_dict(),
                            hist=hist, best=best, best_step=best_step, best_state=best_state, args=vars(a),
                            cool=dict(cool), stale=stale, lm=(lm_order, lm_pos), bal=dict(bal)), f"{d}/loop.pt")
            with open(f"{ck_root}/LATEST", "w") as f:
                f.write(os.path.basename(d))
            for old in os.listdir(ck_root):  # keep only the newest checkpoint (~320 GB each at 30B)
                if old.startswith("step") and old != os.path.basename(d):
                    shutil.rmtree(f"{ck_root}/{old}", ignore_errors=True)
        dist.barrier()
        say(f"checkpoint {d} ({time.time() - t:.0f}s)")

    # --patience: the best model so far (by val loss), model + latents + head only (~115 GB at 30B), replaced at
    # every new best; the end of the run loads it back before the export and the test pass
    keep_best = full and a.patience > 0 and a.schedule != "wsd"  # wsd: the model after the final cooldown is exported
    best_dir = f"{ck_root}/best"

    def save_best(s):
        from torch.distributed.checkpoint.state_dict import get_model_state_dict
        t = time.time()
        if main_proc and os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        dist.barrier()
        dcp.save({"model": get_model_state_dict(holder)}, checkpoint_id=best_dir)
        if main_proc:
            with open(f"{best_dir}/STEP", "w") as f:
                f.write(str(s))
        dist.barrier()
        say(f"best model so far (step {s}) saved to {best_dir} ({time.time() - t:.0f}s)")

    def load_best():
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
        sd = {"model": get_model_state_dict(holder)}
        dcp.load(sd, checkpoint_id=best_dir)
        set_model_state_dict(holder, sd["model"])
        say(f"loaded the best model (step {open(f'{best_dir}/STEP').read().strip()}) from {best_dir}")

    def export_model(path):
        """The sharded Glimmer as it is now -> HF format (bf16) at path; every rank calls it."""
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        t = time.time()
        sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if main_proc:  # fp32 master weights are saved in bf16 (the released dtype), one tensor at a time
            for kk in list(sd):  # in place: never an fp32 and a bf16 copy of the whole model at once
                if sd[kk].is_floating_point():
                    sd[kk] = sd[kk].to(torch.bfloat16)
            model.save_pretrained(path, state_dict=sd, max_shard_size="5GB")
            proc.save_pretrained(path)
        del sd
        dist.barrier()
        say(f"trained Glimmer saved to {path} ({time.time() - t:.0f}s)")

    milestones = []
    for x in a.export_at:
        milestones.append(time.mktime(time.strptime(x, "%Y-%m-%dT%H:%M")))
    milestones.sort()

    def load_ckpt(d):
        """Everything save_ckpt wrote -> the model, optimizer, schedule and loop state (returned as a dict)."""
        msd, osd = get_state_dict(holder, opt)
        sd = {"model": msd, "optim": osd}
        dcp.load(sd, checkpoint_id=d)
        set_state_dict(holder, opt, model_state_dict=sd["model"], optim_state_dict=sd["optim"])
        L = torch.load(f"{d}/loop.pt", weights_only=False)
        for key in ("epochs", "bs", "mb", "k", "lr_full", "lr_head", "max_train", "schedule"):
            assert L["args"].get(key, "cosine" if key == "schedule" else None) == getattr(a, key), \
                f"--resume with a different --{key}: {L['args'].get(key)} vs {getattr(a, key)}"
        rng.bit_generator.state = L["rng"]
        cool.clear()
        cool.update(L.get("cool", {}))
        sched.load_state_dict(L["sched"])
        return L

    start = 0
    stale, lm_order, lm_pos = 0, None, 0
    bal = dict(lat=None, lm=None, n=0, w=a.lm_weight)  # --lm-balance: EMA gradient norms of the two objectives
    if a.resume:
        d = f"{ck_root}/{open(f'{ck_root}/LATEST').read().strip()}"
        L = load_ckpt(d)
        start, order, pos = L["step"], L["order"], L["pos"]
        hist, best, best_step, best_state = L["hist"], L["best"], L["best_step"], L["best_state"]
        stale = L.get("stale", 0)
        lm_order, lm_pos = L.get("lm", (None, 0))
        say(f"resumed from {d}: step {start}/{steps}, best val {best:.4f} at step {best_step}"
            + (f"; in a {cool['kind']} cooldown from step {cool['c0']} ({cool['n']} steps)" if cool else ""))
    else:
        hist = [dict(step=0, val=evaluate(va), **({"lm_val": lm_evaluate()} if lm_items else {}))]
        best, best_step = hist[0]["val"], 0
        best_state = snap()
        order, pos = rng.permutation(tr), 0
        say(f"{a.mode}: {steps} steps (bs {a.bs}, mb {a.mb}, warmup {warm}); step 0 val {best:.4f}"
            + (f", language val {hist[0]['lm_val']:.4f}" if lm_items else ""))
    if lm_order is None:
        lm_order, lm_pos = rng.permutation(len(lm_items.get("train", []))), 0
    if a.resume and L.get("bal"):
        bal.update(L["bal"])
    end_t = job_end() if full and main_proc else None
    if end_t:
        say(f"job ends {time.ctime(end_t)}; checkpoint every {a.ckpt_every_min} min, "
            f"stop {a.stop_margin_min} min before")
    model.train()
    head.train()
    t1, last_ck = time.time(), time.time()
    n_skip = 0
    s, cool_dir, final_done = start, None, False
    if cool:  # resumed inside a cooldown: its return point is the checkpoint it resumed from
        cool_dir = d

    def begin_cooldown(kind, n):
        """wsd: checkpoint here (the constant-LR state to come back to / to continue training from), then cool down."""
        nonlocal cool_dir, last_ck
        save_ckpt(s, order, pos)
        last_ck = time.time()
        cool.update(c0=s, n=n, kind=kind)
        cool_dir = f"{ck_root}/step{s:06d}"
        say(f"{kind} cooldown: steps {s + 1}-{s + n} ((1-sqrt) to lr 0), from the checkpoint {cool_dir}")

    while s < steps:
        s += 1
        if pos + a.bs > len(order):
            order, pos = rng.permutation(tr), 0
        gb = order[pos:pos + a.bs]
        pos += a.bs
        mine = shard(gb)
        l_lm, lm_on = None, bool(lm_items) and s % a.lm_every == 0
        if lm_on:  # one text per process; FSDP averages the gradient over the processes
            if lm_pos + world > len(lm_order):
                lm_order, lm_pos = rng.permutation(len(lm_items["train"])), 0
            lm_item = lm_items["train"][lm_order[lm_pos + rank]]
            lm_pos += world
            if a.lm_balance:  # until both norms are measured: a weight too small to matter (measurement only)
                bal["w"] = max(a.lm_balance * bal["lat"] / bal["lm"], 1e-8) if bal["lat"] and bal["lm"] else 1e-8
            lm_first = bool(a.lm_balance) and bal["n"] % 2 == 0  # this step measures the language gradient alone
        ema = lambda old, new: new if old is None else 0.9 * old + 0.1 * new
        if lm_on and lm_first:
            l_lm = lm_loss(lm_item)
            (bal["w"] * l_lm).backward()
            bal["lm"] = ema(bal["lm"], grad_norm_now() / bal["w"])
        for mb in np.array_split(mine, max(1, len(mine) // a.mb)):
            l, _, _, _ = step_loss(mb, True)
            (l * len(mb) / len(gb) * (world if ddp else 1)).backward()
        if lm_on and not lm_first:
            if a.lm_balance:  # this step measures the VLA-JEPA gradient alone
                bal["lat"] = ema(bal["lat"], grad_norm_now())
            l_lm = lm_loss(lm_item)
            (bal["w"] * l_lm).backward()
        if lm_on and a.lm_balance:
            bal["n"] += 1
        if ddp:
            for p in small:
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad /= world
        gn_s = float(torch.nn.utils.clip_grad_norm_(small, 1.0))
        gn_f = 0.0
        if full_params:
            g = torch.nn.utils.clip_grad_norm_(full_params, 1.0)
            gn_f = float(g.full_tensor() if hasattr(g, "full_tensor") else g)
        # a non-finite loss or gradient on any rank: skip this update everywhere (never write NaN into the weights),
        # name the first bad parameters once, stop when more than --max-skips of the steps are skipped
        bad = torch.tensor([float(not (math.isfinite(float(l)) and math.isfinite(gn_s) and math.isfinite(gn_f)
                                       and (l_lm is None or math.isfinite(float(l_lm)))))], device=dev)
        if ddp:
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if bad.item():
            n_skip += 1
            if n_skip == 1:
                names = []
                for nm, p in model.named_parameters():
                    if p.grad is not None and not torch.isfinite(local(p.grad)).all():
                        names.append(nm)
                        if len(names) == 8:
                            break
                say(f"step {s}: non-finite (loss {float(l):.4g}, grad norm full {gn_f:.4g} / head {gn_s:.4g}); "
                    f"first bad grads on rank {rank}: {names}")
            opt.zero_grad(set_to_none=True)
            sched.step()
            if n_skip > a.max_skips * (s - start):
                raise RuntimeError(f"{n_skip} of {s - start} steps non-finite: stopping")
            continue
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if s - start <= 5:
            say(f"step {s}: loss {float(l):.4f}" + (f", language {float(l_lm):.4f}" if l_lm is not None else "")
                + (f", Glimmer gradient norms (EMA) VLA-JEPA {bal['lat'] or 0:.3g} / language {bal['lm'] or 0:.3g}, "
                   f"language weight {bal['w']:.3g}" if a.lm_balance else "")
                + f", grad norm full {gn_f:.3g} / head {gn_s:.3g}, "
                f"lr {sched.get_last_lr()[-1]:.2e}, {time.time() - t1:.0f} s, "
                f"GPU peak {torch.cuda.max_memory_allocated(dev) / 2 ** 30:.1f} GiB")
            if full_params:  # check the updated weights themselves in the first steps
                nb = []
                for nm, p in inner.named_parameters():
                    if not torch.isfinite(local(p)).all():
                        nb.append(nm)
                        if len(nb) == 8:
                            break
                if nb:
                    raise RuntimeError(f"non-finite weights after step {s} on rank {rank}: {nb}")
        if s % 25 == 0:
            say(f"step {s}/{steps} loss {float(l):.4f}" + (f" language {float(l_lm):.4f}" if l_lm is not None else "")
                + (f" | gradient VLA-JEPA {bal['lat'] or 0:.3g} language {bal['lm'] or 0:.3g} weight {bal['w']:.3g}"
                   if a.lm_balance else "")
                + f" | clipped from {gn_f:.3g} ({(time.time() - t1) / (s - start):.1f} s/step)")
        early = start_final = False
        cool_end = bool(cool) and s >= cool["c0"] + cool["n"]
        if s % every == 0 or s == steps or cool_end:
            vl = evaluate(va)
            phase = f"{cool['kind']} cooldown" if cool else "train"
            row = dict(step=s, val=vl, sec=round(time.time() - t1), phase=phase)
            if lm_items:
                row["lm_val"] = lm_evaluate()
            if a.lm_balance:
                row.update(grad_vla=bal["lat"], grad_lm=bal["lm"], lm_weight=bal["w"])
            hist.append(row)
            say(f"step {s} val {vl:.4f}" + (f" ({hist[-1]['phase']})" if cool else "")
                + (f" | language val {hist[-1]['lm_val']:.4f} (step 0: {hist[0]['lm_val']:.4f})" if lm_items else ""))
            if not cool:  # best / patience on the main curve (constant LR for wsd), never on a cooldown
                if vl < best:
                    best, best_step, best_state, stale = vl, s, snap(), 0
                    if keep_best:
                        save_best(s)
                else:
                    stale += 1
                    if a.patience and stale >= a.patience:
                        say(f"{'final cooldown' if wsd else 'early stop'} at step {s}/{steps}: {stale} validations "
                            f"without a new best (best {best:.4f} at step {best_step})")
                        early, start_final = not wsd, wsd
            if full and milestones and not wsd:  # rank 0's clock decides, so every rank exports at the same step
                due = torch.tensor([float(main_proc and time.time() >= milestones[0])], device=dev)
                dist.broadcast(due, 0)
                if due.item():
                    milestones.pop(0)
                    md = f"{od}/glimmer_step{s:06d}"
                    export_model(md)
                    if main_proc:
                        torch.save(snap(), f"{md}/trainable.pt")
                        json.dump(dict(step=s, epoch=round(s * a.bs / len(tr), 2), val=vl, best_val=best,
                                       best_step=best_step, is_best=best_step == s, hist=hist),
                                  open(f"{md}/milestone.json", "w"), indent=1, default=float)
                    say(f"milestone export at step {s}: {md} (val {vl:.4f}; best {best:.4f} at step {best_step})")
        if wsd:
            if cool_end and cool["kind"] == "milestone":  # export the cooled model, then back to the constant LR
                md = f"{od}/glimmer_step{s:06d}"
                export_model(md)
                if main_proc:
                    torch.save(snap(), f"{md}/trainable.pt")
                    json.dump(dict(step=s, epoch=round(s * a.bs / len(tr), 2), val=vl, cooldown=dict(cool),
                                   constant_lr_best_val=best, constant_lr_best_step=best_step, hist=hist),
                              open(f"{md}/milestone.json", "w"), indent=1, default=float)
                say(f"milestone export at step {s}: {md} (val {vl:.4f} after a {cool['n']}-step cooldown; "
                    f"constant-LR best {best:.4f})")
                L = load_ckpt(cool_dir)
                s, order, pos = L["step"], L["order"], L["pos"]
                lm_order, lm_pos = L.get("lm", (lm_order, lm_pos))
                say(f"back to step {s} at the constant LR (reloaded {cool_dir})")
                continue
            if cool_end and cool["kind"] == "final":
                final_done, early = True, True
            elif not cool and s < steps:
                due = torch.zeros(2, device=dev)
                if main_proc:
                    sec_step = (time.time() - t1) / max(1, s - start)
                    n_m = math.ceil(a.milestone_cooldown_frac * s)
                    # start so that the export lands at the milestone: cooldown + ~45 min (validation, export, reload)
                    due[0] = float(bool(milestones) and s >= warm
                                   and time.time() + n_m * sec_step + 2700 >= milestones[0])
                    due[1] = float(s >= steps - math.ceil(a.cooldown_frac * steps))
                dist.broadcast(due, 0)
                if start_final or due[1].item():
                    begin_cooldown("final", max(1, min(math.ceil(a.cooldown_frac * s), steps - s)))
                elif due[0].item():
                    milestones.pop(0)
                    begin_cooldown("milestone", max(1, min(math.ceil(a.milestone_cooldown_frac * s), steps - s)))
        if early:
            break
        if full and s < steps:  # rank 0 decides, so every rank checkpoints / stops at the same step
            act = torch.zeros(1, device=dev)
            if main_proc:
                if a.ckpt_every_min and time.time() - last_ck > a.ckpt_every_min * 60:
                    act[0] = 1
                if (a.pause_after_steps and s - start >= a.pause_after_steps) or \
                        (end_t and end_t - time.time() < a.stop_margin_min * 60):
                    act[0] = 3 if a.finish_at_limit else 2
            dist.broadcast(act, 0)
            if cool and act.item() == 1:
                act[0] = 0  # no periodic checkpoints inside a cooldown (its start is already on disk)
            if cool and act.item() == 2:  # resuming restarts this cooldown from its start (LATEST)
                say(f"stopped at step {s}/{steps} inside a {cool['kind']} cooldown; --resume restarts it from "
                    f"{cool_dir}")
                dist.destroy_process_group()
                return
            if act.item() == 3:
                say(f"job time limit near at step {s}/{steps}: ending the training here (--finish-at-limit)")
                break
            if act.item():
                save_ckpt(s, order, pos)
                last_ck = time.time()
            if act.item() == 2:
                say(f"stopped at step {s}/{steps}; continue with the same arguments + --resume")
                dist.destroy_process_group()
                return
    last = s
    keep = full and (a.save_model == 1 or (a.save_model == -1 and not a.max_train))
    if keep and not final_done:  # the last training state (weights + optimizer + loop): to continue training later
        save_ckpt(last, order, pos)  # (--resume), and insurance for the export below; wsd after its final cooldown:
    #                                 the checkpoint at the cooldown's start is that state (constant LR, to go on from)
    if keep_best and best_step and best_step != last:  # export and test the best model, not the last one
        load_best()
    if full:  # the sharded model is the best one (--patience) or the last; the latents + head match it
        best_state = snap()
        if keep:
            export_model(f"{od}/glimmer")
    with torch.no_grad():
        latents.copy_(best_state["latents"].to(dev))
        head.load_state_dict(best_state["head"])
        for p, q in zip(lora_params, best_state["lora"]):
            p.copy_(q.to(p.device))
    if main_proc:
        torch.save(best_state, f"{od}/trainable.pt")

    # ---------------------------------------------------------------- final pass: test predictions, latents for probes
    model.eval()
    head.eval()
    allidx = np.concatenate([tr0, te])
    P, Z, Hm = {}, {}, {}
    te_set = set()
    for i in te:
        te_set.add(int(i))
    with torch.no_grad():
        for b, dummy in batches(allidx):
            _, p, z, h = step_loss(b, False)
            if dummy:
                continue
            for i, j in enumerate(b):
                if int(j) in te_set:
                    P[int(j)] = p[i].float().cpu().numpy()
                Z[int(j)], Hm[int(j)] = z[i].float().cpu().numpy(), h[i].float().mean(0).cpu().half().numpy()
    if ddp:
        parts = [None] * world
        dist.all_gather_object(parts, (P, Z, Hm))
        for pp, zz, hh in parts:
            P.update(pp)
            Z.update(zz)
            Hm.update(hh)
    if not main_proc:
        dist.destroy_process_group()
        return
    Yn, Cn = Y.numpy(), C.numpy()
    res = {"mode": a.mode, "vision_lora": a.vision_lora, "k": k, "fold": 0, "history": hist, "best_val": best,
           "best_step": best_step, "steps": steps, "last_step": last, "minutes": round((time.time() - t0) / 60, 1),
           "args": vars(a)}
    for j, hz in enumerate(horizons):
        ok = te[valid[te, j]]
        Pj = []
        for i in ok:
            Pj.append(P[int(i)][j])
        Pj = np.stack(Pj)
        res[f"r2_delta_{hz}"] = r2_vw(Yn[ok, j] - Cn[ok], Pj - Cn[ok])
        res[f"r2_abs_{hz}"] = r2_vw(Yn[ok, j], Pj)
    res["probe_z"] = probes(stack_rows(Z, tr0), stack_rows(Z, te), lab.iloc[tr0], lab.iloc[te])
    from sklearn.decomposition import PCA
    htr, hte = stack_rows(Hm, tr0).astype(np.float32), stack_rows(Hm, te).astype(np.float32)
    pca = PCA(min(256, len(htr) - 1), random_state=0).fit(htr)
    res["probe_latent_hidden"] = probes(pca.transform(htr), pca.transform(hte), lab.iloc[tr0], lab.iloc[te])
    json.dump(res, open(f"{od}/result.json", "w"), indent=1, default=float)
    summary = {}
    for kk, vv in res.items():
        if kk not in ("history", "args"):
            summary[kk] = vv
    say(summary)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
