"""Human demonstration data: play index, step/clip layout, feature loading.

Step layout (identical for humans and the agent):
  decision step t starts at emulator frame f = FRAME_SKIP * t;
  observation o_t = screens [f-15, f-13, ..., f-1] (CLIP_LEN=8, stride 2), i.e. only what
  was on screen before the step's input; frames before 0 are clamped to the first screen;
  action a_t = button state on frame f, projected to mjg.actions.COMBOS;
  ram_t = RAM variables after frame f-1 (clamped to 0).
Only odd frames are ever needed for clips, so replays keep odd screens only.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np

from .actions import FRAME_SKIP

CLIP_LEN = 8
CLIP_STRIDE = 2

# SMB player_state (RAM 0x000E): 8 = normal control; 9 = changing size, 10 = injury
# blink, 12 = getting fire flower (player still steers). Others are cut-scenes/deaths.
CONTROL_STATES = (8, 9, 10, 12)

_PAT = re.compile(r"(sub-\d+)_(ses-\d+)_task-mario_level-(w\dl\d)_rep-(\d+)\.bk2$")


@dataclass
class Play:
    play_id: str
    sub: str
    ses: str
    level: str
    rep: int
    bk2: str


def index_plays(data_root: str, subjects=None) -> list[Play]:
    """Every non-empty .bk2 replay under <data_root>/sub-*/ses-*/, sorted by path (optionally only ``subjects``)."""
    out = []
    for p in sorted(glob.glob(os.path.join(data_root, "sub-*", "ses-*", "*.bk2"))):
        m = _PAT.search(os.path.basename(p))
        if not m or os.path.getsize(p) == 0:
            continue
        sub, ses, level, rep = m.groups()
        if subjects and sub not in subjects:
            continue
        out.append(Play(f"{sub}_{ses}_{level}_rep-{rep}", sub, ses, level, int(rep), p))
    return out


def clip_positions(n_steps: int, n_odd: int) -> np.ndarray:
    """Indices into the odd-frame array for every step's clip: [n_steps, CLIP_LEN].

    Odd frame k sits at position (k-1)//2. For step t (f=4t) the clip frames are
    f-1-2j (j = CLIP_LEN-1..0) -> positions 2t-1-j.
    """
    t = np.arange(n_steps)[:, None]
    j = np.arange(CLIP_LEN - 1, -1, -1)[None, :]
    pos = (FRAME_SKIP // 2) * t - 1 - j * (CLIP_STRIDE // 2)
    return np.clip(pos, 0, n_odd - 1)


def load_split(feat_dir: str, play_ids: list[str], mmap: bool = True):
    """Load per-step features (<play>.z.npy) and metadata (<play>.meta.npz) for the given plays.

    Returns (zs, metas): one array per play (memory-mapped by default, so plays are not all held
    in RAM) and one dict of metadata arrays per play, in the order of ``play_ids``.
    """
    zs, metas = [], []
    for pid in play_ids:
        zs.append(np.load(os.path.join(feat_dir, pid + ".z.npy"), mmap_mode="r" if mmap else None))
        metas.append(dict(np.load(os.path.join(feat_dir, pid + ".meta.npz"))))
    return zs, metas


def read_index(feat_dir: str) -> list[dict]:
    """Every <play>.meta.json of a feature folder, sorted by file name."""
    rows = []
    for p in sorted(glob.glob(os.path.join(feat_dir, "*.meta.json"))):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


# ---------------------------------------------------------------------------------------------------
# V-JEPA 2.1 G features (features/vjepa21G_c16s8_f8_g8, scripts/features/encode_frames.py)
#   <play>.npy [rows, 4 depths, 64 patches, 1664] float16; row r = 2 s clip ending at screen 8r.
# Row masks from scripts/features/build_row_masks.py live in the sibling folder <feat_dir>_rowmask/.
# By default rows outside play mode (title screen, attract demo, game over) are SKIPPED.
# ---------------------------------------------------------------------------------------------------
VJEPA_ROWS = ("gameplay", "in_control", "all")
_MASK_CACHE: dict = {}


def vjepa_row_mask(feat_dir: str, play_id: str, rows: str = "gameplay") -> np.ndarray:
    """Boolean [n_rows] for one play. rows: 'gameplay' (default: whole clip in play mode),
    'in_control' (gameplay + Mario in normal control, timer running) or 'all'."""
    if rows not in VJEPA_ROWS:
        raise ValueError(f"rows must be one of {VJEPA_ROWS}")
    if rows == "all":
        with open(os.path.join(feat_dir, play_id + ".json")) as f:
            return np.ones(json.load(f)["shape"][0], bool)
    path = os.path.join(feat_dir.rstrip("/") + "_rowmask", "masks.npz")
    if path not in _MASK_CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: run scripts/features/build_row_masks.py, or pass rows='all'")
        _MASK_CACHE[path] = np.load(path)
    return _MASK_CACHE[path][f"{play_id}/{rows}"]


def load_vjepa_play(feat_dir: str, play_id: str, rows: str = "gameplay", depth: int | None = None,
                    last: bool = False):
    """Features of one play with masked rows removed.

    Returns (x, row_idx, screens):
      x        [n_kept, 4, 64, 1664] (or [n_kept, 64, 1664] with depth=0..3 index into blocks
               [11, 23, 37, 47], or the .last.npy file with last=True), float16, in memory
      row_idx  indices of the kept rows in the full file
      screens  play screen (mp4 frame) at which each kept row's clip ends
    """
    keep = np.where(vjepa_row_mask(feat_dir, play_id, rows))[0]
    a = np.load(os.path.join(feat_dir, play_id + (".last.npy" if last else ".npy")), mmap_mode="r")
    x = a[keep] if (depth is None or last) else a[keep, depth]
    screens = np.load(os.path.join(feat_dir, play_id + ".game.npz"))["frame_idx"][keep]
    return np.asarray(x), keep, screens
