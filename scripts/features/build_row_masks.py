"""Row masks for the V-JEPA 2.1 G features: which rows are gameplay and which have Mario in control.

The feature files are left untouched. This writes a sibling folder <feat_dir>_rowmask/ (outside the
feature folder, so scripts that glob <feat_dir>/*.json are not affected) holding masks.npz, with two
boolean arrays per play (one value per feature row):

  <play>/gameplay    the WHOLE 2 s clip of the row (screens k-120..k, every 8th, clamped at 0) is in
                     play mode (RAM 0x770 oper_mode == 1). Drops the title screen, the attract demo
                     shown after a game over, the game-over screen, and the ~2 s after play resumes
                     (those clips still contain title / demo images).
  <play>/in_control  gameplay, and at the row's screen k Mario is in normal control
                     (player_state 0x0E == 8) with the game timer running (timer lower than 1.5 s
                     earlier: not a level-start / respawn freeze, death animation or lives screen).
                     Same rule as scripts/data/build_moment_index.py, without its 3 s-into-the-life minimum.

plus summary.json (definitions and counts). Read with mjg.data.load_vjepa_play, which
skips non-gameplay rows by default.

  python scripts/features/build_row_masks.py --feat-dir $MJG_ROOT/features/vjepa21G_c16s8_f8_g8 [--check 20]

--check N compares the timer from <play>.game.npz (ram_time) with the timer digits of the full RAM
(features/ram_full, as used by the moment index) on N plays, and exits on any mismatch.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

CLIP_LEN, CLIP_STRIDE = 16, 8        # V-JEPA clip: 16 screens, every 8th (from the feature json)
TIMER_LAG = 90                       # timer running = lower than 1.5 s earlier (moment index rule)


def clip_screens(k: np.ndarray, clip_len: int = CLIP_LEN, stride: int = CLIP_STRIDE) -> np.ndarray:
    """[rows, clip_len] screens of each row's clip, oldest first, clamped at 0 (as in encode_frames.py)."""
    j = np.arange(clip_len - 1, -1, -1)[None, :] * stride
    return np.clip(k[:, None] - j, 0, None)


def play_masks(g, clip_len: int, stride: int):
    k = g["frame_idx"].astype(int)
    om = g["ram_oper_mode"].astype(int)
    st = g["ram_player_state"].astype(int)
    t = g["ram_time"].astype(int)
    gameplay = (om[clip_screens(k, clip_len, stride)] == 1).all(1)
    running = np.zeros(len(t), bool)
    running[TIMER_LAG:] = t[TIMER_LAG:] < t[:-TIMER_LAG]
    in_control = gameplay & (st[k] == 8) & running[k]
    return gameplay, in_control


def check_timer(feat_dir: str, ram_dir: str, plays: list[str]):
    for p in plays:
        g = np.load(os.path.join(feat_dir, p + ".game.npz"))
        ram = np.load(os.path.join(ram_dir, p + ".npz"))["ram"]
        d = ram[:, 0x7F8:0x7FB].astype(int)
        full = d[:, 0] * 100 + d[:, 1] * 10 + d[:, 2]
        if len(full) != len(g["ram_time"]) or not np.array_equal(full, g["ram_time"].astype(int)):
            raise SystemExit(f"timer mismatch on {p}: game.npz ram_time != full-RAM digits")
    print(f"timer check ok on {len(plays)} plays")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--ram-dir", default=None, help="features/ram_full (for --check)")
    ap.add_argument("--check", type=int, default=20)
    a = ap.parse_args()
    ram_dir = a.ram_dir or os.path.join(os.path.dirname(a.feat_dir.rstrip("/")), "ram_full")
    metas = sorted(glob.glob(os.path.join(a.feat_dir, "*.json")))
    out_dir = a.feat_dir.rstrip("/") + "_rowmask"
    os.makedirs(out_dir, exist_ok=True)
    plays = []
    for path in metas:
        plays.append(os.path.basename(path)[:-5])
    if a.check:
        rng = np.random.default_rng(0)
        check_timer(a.feat_dir, ram_dir, list(rng.choice(plays, min(a.check, len(plays)), replace=False)))
    t0 = time.time()
    out, per_play = {}, {}
    tot = dict(rows=0, gameplay=0, in_control=0)
    for m, p in zip(metas, plays):
        meta = json.load(open(m))
        g = np.load(os.path.join(a.feat_dir, p + ".game.npz"))
        gp, ic = play_masks(g, meta.get("clip_len", CLIP_LEN), meta.get("clip_stride", CLIP_STRIDE))
        if len(gp) != meta["shape"][0]:
            raise SystemExit(f"{p}: {len(gp)} mask rows but {meta['shape'][0]} feature rows")
        out[f"{p}/gameplay"], out[f"{p}/in_control"] = gp, ic
        per_play[p] = dict(rows=len(gp), gameplay=int(gp.sum()), in_control=int(ic.sum()))
        for kk, v in (("rows", len(gp)), ("gameplay", gp.sum()), ("in_control", ic.sum())):
            tot[kk] += int(v)
    np.savez_compressed(os.path.join(out_dir, "masks.npz"), **out)
    no_gameplay = []
    for p, v in per_play.items():
        if v["gameplay"] == 0:
            no_gameplay.append(p)
    definitions = {
        "gameplay": "whole 2 s clip (screens k-120..k every 8, clamped at 0) has oper_mode (0x770) == 1",
        "in_control": "gameplay and player_state (0x0E) == 8 at k and game timer at k lower than "
                      f"{TIMER_LAG} screens earlier"}
    summary = dict(
        created=time.strftime("%Y-%m-%d"), script="scripts/features/build_row_masks.py", definitions=definitions,
        totals=tot, share_gameplay=round(tot["gameplay"] / tot["rows"], 4),
        share_in_control=round(tot["in_control"] / tot["rows"], 4),
        plays_without_gameplay=no_gameplay, per_play=per_play)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(f"{len(plays)} plays in {time.time() - t0:.0f} s; rows {tot['rows']}, gameplay {tot['gameplay']} "
          f"({100 * summary['share_gameplay']:.1f}%), in control {tot['in_control']} "
          f"({100 * summary['share_in_control']:.1f}%); plays with no gameplay rows: "
          f"{len(summary['plays_without_gameplay'])}")


if __name__ == "__main__":
    main()
