"""Per-TR features of one continuous Super Mario Bros. play (an agent's or a person's), as the encoders expect them.

The definitions are the ones the encoders were trained with (brain_pred/encoding/build_tr_features.py, applied to
the human fMRI runs): the same `gamestate` function and the same button summary, binned in TRs of 1.49 s.
  buttons    [12] per button A, B, LEFT, RIGHT, UP, DOWN: share of frames pressed, presses starting in the TR / 10
  gamestate  [13] RAM-derived (Mario speed / pose / position, enemies), mean over the TR
  t_play, level   time since the play started, level name (nuisance band)
  lowlevel   [3]  luminance, optical flow, audio envelope, mean over the TR (nuisance band of --lowlevel encoders;
                  per frame from brain_pred/features/lowlevel.py, e.g. the replay's `lowlevel`)
  vjepa      [4, 1664] V-JEPA 2.1 G, optional: computed separately on the play's frames (GPU), as for the human runs
  agent      [4, hidden] the agent's own internal state (glimmer_play --save-hidden), per TR: the state at its most
             recent decision before the TR ends, the same rule as the calibration on the human scans
             (brain_pred/agent/embed_scans.py)
Frame 0 of the inputs is the start of the play.
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "encoding"))
from build_tr_features import gamestate as ram_gamestate, BTN  # noqa: E402  (same code as training)

TR, FPS = 1.49, 60


def tr_features(buttons, ram, button_names, level, tr=TR, fps=FPS, lowlevel=None):
    """buttons [n_frames, n_buttons] bool, ram [n_frames, 2048] (RAM after each frame's input), optional lowlevel
    [n_frames, 3] (same frames) -> dict of per-TR bands."""
    n = min(len(buttons), len(ram))
    T = int(n // (tr * fps))
    names = list(button_names)
    col = []
    for b in BTN:
        col.append(names.index(b))
    gsf = ram_gamestate(np.asarray(ram[:n]))
    bt, gs, ll = np.zeros((T, 12), np.float32), np.zeros((T, 13), np.float32), np.zeros((T, 3), np.float32)
    for t in range(T):
        f0, f1 = int(round(t * tr * fps)), int(round((t + 1) * tr * fps))
        b = np.asarray(buttons[f0:f1])[:, col]
        prev = np.asarray(buttons[f0 - 1])[col] if f0 else np.zeros(6, bool)
        on = np.diff(np.vstack([prev[None], b]).astype(int), axis=0) == 1  # presses starting in the TR
        bt[t] = np.r_[b.mean(0), on.sum(0) / 10]
        gs[t] = gsf[f0:f1].mean(0)
        if lowlevel is not None:
            ll[t] = np.asarray(lowlevel[f0:f1]).mean(0)
    out = dict(buttons=bt, gamestate=gs, t_play=(np.arange(T) * tr).astype(np.float32), level=np.array([level] * T))
    if lowlevel is not None:
        out["lowlevel"] = ll
    return out


def load_play_meta(play_dir):
    """A glimmer_play / odysseus play folder -> (inputs [n_frames, n_buttons] bool, button_names, level)."""
    meta = json.load(open(f"{play_dir}/meta.json"))
    return np.load(f"{play_dir}/inputs.npy"), meta["button_names"], meta["level"]


def turn_starts(play_dir):
    """Frame at which each turn of a glimmer_play play starts (after the level intro; hold + one NOOP frame)."""
    meta = json.load(open(f"{play_dir}/meta.json"))
    f, out = int(meta.get("start_frames", 0)), []
    for line in open(f"{play_dir}/steps.jsonl"):
        out.append(f)
        h = int(json.loads(line).get("frames_held", 1))
        f += h + (1 if h > 1 else 0)
    return np.array(out)


def agent_tr_features(hidden, starts, n_tr, tr=TR, fps=FPS):
    """hidden [n_turns, 4, H] + turn start frames -> [n_tr, 4, H]: state of the last turn starting before each TR ends.
    TRs before the first decision get the first state."""
    ends = (np.arange(n_tr) + 1) * tr * fps
    idx = np.clip(np.searchsorted(starts, ends, side="left") - 1, 0, len(hidden) - 1)
    return np.asarray(hidden, np.float32)[idx]
