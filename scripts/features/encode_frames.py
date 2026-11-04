"""Dense frozen-V-JEPA features: one clip per screen (or every ``--frame-step`` screens) of every play,
at several network depths.

Row r of a play = the clip of ``--clip-len`` screens ending at screen k = r * ``--frame-step`` (inclusive),
taken every ``--clip-stride`` frames: screens k-14, k-12, ..., k for the defaults. Screen k is the image after
the k-th input, i.e. mp4 frame k (docs/DATA.md); clips before the first screen repeat screen 0.
No screen is skipped: level intros, deaths and transitions are encoded too (see build_row_masks.py).

For V-JEPA 2.1 (``vjepa2.1:<folder>``) each clip yields the 4 blocks Meta's model exposes
(11, 23, 37, 47 for ViT-G; 47 is the usual final output), each after its own LayerNorm, averaged
over the clip's time and pooled to a ``--grid`` x ``--grid`` map. A Hugging Face V-JEPA 2 gives its
final layer only. Output per play, written atomically (plays already done are skipped):

  <out>/<play_id>.npy        float16 [n_rows, n_depths, grid*grid, D]
  <out>/<play_id>.last.npy   (--save-last) float16 [n_rows, grid*grid, D]: final block, most recent tubelet only
  <out>/<play_id>.game.npz   (--save-game) row screens (frame_idx), buttons and RAM variables of every frame
  <out>/<play_id>.json       shapes, depths, clip definition, encoder, Meta code commit, timing

Plays are split across ``--nshards`` jobs in contiguous blocks of about equal frame counts, in
subject order, so the first jobs finish sub-01 first. Each job stops before its output folder
exceeds ``--max-tb`` terabytes.

python scripts/features/encode_frames.py --subjects sub-01 --out $MJG_ROOT/features/vjepa21G_frames \
    --encoder 'vjepa2.1:${MJG_VJEPA21_G}' --shard 0 --nshards 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mjg import data as D  # noqa: E402
from mjg import emulator as E  # noqa: E402

SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]


def folder_bytes(path):
    total = 0
    for e in os.scandir(path):
        if e.is_file():
            total += e.stat().st_size
    return total


def shard_blocks(plays, n_frames, nshards):
    """Contiguous blocks of plays with about equal total frames."""
    target, blocks, cur, acc = sum(n_frames) / nshards, [], [], 0
    for p, n in zip(plays, n_frames):
        cur.append(p)
        acc += n
        if acc >= target * (len(blocks) + 1) and len(blocks) < nshards - 1:
            blocks.append(cur)
            cur = []
    blocks.append(cur)
    while len(blocks) < nshards:
        blocks.append([])
    return blocks


def clip_index(n, clip_len, stride, device):
    import torch
    k = torch.arange(n, device=device)[:, None]
    j = torch.arange(clip_len - 1, -1, -1, device=device)[None, :] * stride
    return (k - j).clamp_min(0)  # [n, clip_len], oldest first


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MJG_DATA"))
    ap.add_argument("--integration", default=os.environ.get("MJG_INTEGRATION"))
    ap.add_argument("--subjects", nargs="*", default=SUBJECTS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="vjepa2.1:${MJG_VJEPA21_G}")
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--clip-stride", type=int, default=2)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--plays", nargs="*", help="only these play ids")
    ap.add_argument("--max-tb", type=float, default=4.2, help="stop before the output folder exceeds this")
    ap.add_argument("--frame-step", type=int, default=1,
                    help="one sample every k screens (row r = clip ending at screen r*k)")
    ap.add_argument("--save-last", action="store_true",
                    help="also <play>.last.npy: final block, most recent tubelet only")
    ap.add_argument("--save-game", action="store_true",
                    help="also <play>.game.npz: sampled frame indices, buttons and RAM variables per frame")
    return ap.parse_args()


def plays_to_do(a):
    """This shard's plays that have no output yet."""
    plays = D.index_plays(a.data, a.subjects)
    plays.sort(key=lambda p: (a.subjects.index(p.sub) if p.sub in a.subjects else 99, p.play_id))
    if a.plays:
        wanted = set(a.plays)
        selected = []
        for p in plays:
            if p.play_id in wanted:
                selected.append(p)
        plays = selected
    n_frames = []
    for p in plays:
        n_frames.append(len(E.read_bk2_inputs(p.bk2)[1]))
    mine = shard_blocks(plays, n_frames, a.nshards)[a.shard]
    todo = []
    for p in mine:
        if not os.path.exists(os.path.join(a.out, p.play_id + ".json")):
            todo.append(p)
    return mine, todo


def meta_code_commit(encoder):
    """Commit of Meta's code clone (code_commit.txt in the V-JEPA 2.1 folder), if recorded."""
    if not encoder.startswith("vjepa2.1:"):
        return None
    cpath = os.path.join(os.path.expandvars(encoder[len("vjepa2.1:"):]), "code_commit.txt")
    if not os.path.exists(cpath):
        return None
    with open(cpath) as fh:
        return fh.read().strip()


def all_finite(z, m):
    """Finite-value check on about 50 rows spread over the play."""
    return bool(np.isfinite(z[:: max(1, m // 50)]).all())


def save_game(a, p, rep, ends):
    names = []
    for b in rep["button_names"]:
        names.append(str(b))
    ram = {}
    for k, v in rep["ram"].items():
        ram["ram_" + k] = np.asarray(v)
    np.savez(os.path.join(a.out, p.play_id + ".game.npz"), frame_idx=ends.cpu().numpy(),
             buttons=rep["buttons"], button_names=np.array(names), **ram)


def play_meta(a, p, enc, depths, commit, n, m, dt):
    grid2 = a.grid * a.grid
    return {
        "play_id": p.play_id, "sub": p.sub, "ses": p.ses, "level": p.level, "rep": p.rep,
        "n_screens": n, "shape": [m, len(depths), grid2, enc.dim], "dtype": "float16",
        "n_samples": m, "frame_step": a.frame_step, "row_r": f"row r = clip ending at screen k = r*{a.frame_step}",
        "last_shape": [m, grid2, enc.dim] if a.save_last else None,
        "last": "final block, most recent tubelet (last 2 clip frames) only, pooled like the main file"
                if a.save_last else None,
        "game_file": p.play_id + ".game.npz" if a.save_game else None,
        "depths": depths, "grid": a.grid, "clip_len": a.clip_len, "clip_stride": a.clip_stride,
        "row_k": f"clip of screens k-{(a.clip_len - 1) * a.clip_stride}..k every {a.clip_stride} "
                 "(screen k = mp4 frame k)",
        "encoder": enc.model_id, "meta_code_commit": commit, "seconds": round(dt, 1),
        "clips_per_s": round(m / dt, 2)}


def encode_play(a, p, enc, depths, commit, i, n_todo):
    """Replay one play, encode its clips and write its files (nothing when a value is not finite)."""
    import torch
    by_depth = hasattr(enc, "forward_depths")
    t0 = time.time()
    rep = E.replay_bk2(p.bk2, a.integration)
    frames = torch.from_numpy(rep["frames"]).cuda()
    n = len(frames)
    t_rep = time.time() - t0
    ends = torch.arange(0, n, a.frame_step, device=frames.device)
    idx = clip_index(n, a.clip_len, a.clip_stride, frames.device)[ends]
    m = len(ends)
    tmp = os.path.join(a.out, p.play_id + ".npy.tmp")
    z = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                  shape=(m, len(depths), a.grid * a.grid, enc.dim))
    tmp_last = os.path.join(a.out, p.play_id + ".last.npy.tmp")
    zl = (np.lib.format.open_memmap(tmp_last, mode="w+", dtype=np.float16, shape=(m, a.grid * a.grid, enc.dim))
          if a.save_last else None)
    for s in range(0, m, a.batch):
        clips = frames[idx[s:s + a.batch]]  # [b, clip_len, H, W, 3]
        if a.save_last:
            out, last = enc.forward_depths_last(clips)
        else:
            out = enc.forward_depths(clips) if by_depth else enc(clips)[:, None]
            last = None
        z[s:s + len(clips)] = out.cpu().numpy()
        if last is not None:
            zl[s:s + len(clips)] = last.cpu().numpy()
    z.flush()
    ok = all_finite(z, m)
    if zl is not None:
        zl.flush()
        ok = ok and all_finite(zl, m)
    del z, zl
    if not ok:
        os.remove(tmp)
        if a.save_last:
            os.remove(tmp_last)
        print(f"{p.play_id}: NON-FINITE features, skipped", flush=True)
        return
    os.replace(tmp, os.path.join(a.out, p.play_id + ".npy"))
    if a.save_last:
        os.replace(tmp_last, os.path.join(a.out, p.play_id + ".last.npy"))
    if a.save_game:
        save_game(a, p, rep, ends)
    dt = time.time() - t0
    with open(os.path.join(a.out, p.play_id + ".json"), "w") as fh:
        json.dump(play_meta(a, p, enc, depths, commit, n, m, dt), fh, indent=1)
    print(f"[{i + 1}/{n_todo}] {p.play_id}: {m} clips in {dt:.0f} s ({m / dt:.1f}/s, replay {t_rep:.1f} s)",
          flush=True)


def main():
    a = parse_args()
    from mjg.encoder import FrozenVJEPA2

    os.makedirs(a.out, exist_ok=True)
    mine, todo = plays_to_do(a)
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} plays, {len(todo)} to do", flush=True)
    if not todo:
        return
    enc = FrozenVJEPA2(a.encoder, grid=a.grid)
    depths = enc.depths if hasattr(enc, "forward_depths") else ["final"]
    commit = meta_code_commit(a.encoder)
    budget = a.max_tb * 1e12
    for i, p in enumerate(todo):
        used = folder_bytes(a.out)
        if used > budget:
            print(f"STOP: {a.out} holds {used / 1e12:.2f} TB > --max-tb {a.max_tb}", flush=True)
            sys.exit(3)
        encode_play(a, p, enc, depths, commit, i, len(todo))


if __name__ == "__main__":
    main()
