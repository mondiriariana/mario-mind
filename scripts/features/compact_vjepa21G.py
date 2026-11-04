"""Compact copy of the V-JEPA 2.1 G features: grid means and 2x2 quadrants instead of the full 8x8 grids. CPU.

From $MJG_ROOT/features/vjepa21G_c16s8_f8_g8/<play>.npy [n, 4 depths (11, 23, 37, 47), 8x8 grid, 1664] and
<play>.last.npy [n, 8x8, 1664] (scripts/features/encode_frames.py), writes
$MJG_ROOT/features/vjepa21G_c16s8_f8_compact/<play>.npz:
  pooled     float16 [n, 4, 1664]  each depth averaged over the 8x8 grid
  b47_2x2    float16 [n, 4, 1664]  block 47 pooled to 2x2 quadrants (row-major: top-left, top-right, ...)
  last_2x2   float16 [n, 4, 1664]  newest tubelet (block 47) pooled to 2x2; the mean of its quadrants is the
                                   grid mean of <play>.last.npy
and copies <play>.game.npz and <play>.json unchanged. The row masks folder is not touched. Plays already
compacted are skipped.

--verify: for every play, row counts equal the source's, all values finite, and 3 random rows recomputed from
the source match (float16 tolerance). Writes <out>/verify.json. Deletes nothing.

  python scripts/features/compact_vjepa21G.py [--verify] [--procs 16]
"""
import argparse
import glob
import json
import os
import shutil
from multiprocessing import Pool

import numpy as np

ROOT = os.environ.get("MJG_ROOT", ".")
SRC = f"{ROOT}/features/vjepa21G_c16s8_f8_g8"
DST = f"{ROOT}/features/vjepa21G_c16s8_f8_compact"
KEYS = ("pooled", "b47_2x2", "last_2x2")
SIDE_FILES = (".game.npz", ".json")


def q2(x):
    """[..., 64, D] (8x8 row-major) -> [..., 4, D] (2x2 quadrants)."""
    s = x.shape
    return x.reshape(*s[:-2], 2, 4, 2, 4, s[-1]).mean((-4, -2)).reshape(*s[:-2], 4, s[-1])


def compute(z, zl, rows):
    x = np.asarray(z[rows], np.float32)          # [r, 4, 64, D]
    xl = np.asarray(zl[rows], np.float32)        # [r, 64, D]
    return x.mean(2), q2(x[:, 3]), q2(xl)


def load_source(pid):
    z = np.load(f"{SRC}/{pid}.npy", mmap_mode="r")
    zl = np.load(f"{SRC}/{pid}.last.npy", mmap_mode="r")
    return z, zl


def one(pid):
    out = f"{DST}/{pid}.npz"
    if os.path.exists(out):
        return pid, "exists"
    z, zl = load_source(pid)
    n = len(z)
    P, B, L = [], [], []
    for s in range(0, n, 256):
        p, b, l = compute(z, zl, slice(s, min(n, s + 256)))
        P.append(p)
        B.append(b)
        L.append(l)
    np.savez(out + ".tmp.npz", pooled=np.concatenate(P).astype(np.float16),
             b47_2x2=np.concatenate(B).astype(np.float16), last_2x2=np.concatenate(L).astype(np.float16))
    os.replace(out + ".tmp.npz", out)
    for ext in SIDE_FILES:
        if os.path.exists(f"{SRC}/{pid}{ext}"):
            shutil.copy2(f"{SRC}/{pid}{ext}", f"{DST}/{pid}{ext}")
    return pid, f"{n} rows"


def all_finite(c):
    for k in c.files:
        if not np.isfinite(c[k].astype(np.float32)).all():
            return False
    return True


def check(pid):
    try:
        c = np.load(f"{DST}/{pid}.npz")
        z, zl = load_source(pid)
        n = len(z)
        shapes_ok = True
        for k in KEYS:
            if c[k].shape != (n, 4, 1664):
                shapes_ok = False
        fin = all_finite(c)
        rows = np.random.default_rng(hash(pid) % 2**32).choice(n, min(3, n), replace=False)
        p, b, l = compute(z, zl, np.sort(rows))
        err = max(float(np.abs(c["pooled"][np.sort(rows)].astype(np.float32) - p).max()),
                  float(np.abs(c["b47_2x2"][np.sort(rows)].astype(np.float32) - b).max()),
                  float(np.abs(c["last_2x2"][np.sort(rows)].astype(np.float32) - l).max()))
        scale = float(np.abs(p).max()) + 1e-6
        side = True
        for e in SIDE_FILES:
            if not os.path.exists(f"{DST}/{pid}{e}"):
                side = False
                break
        return pid, dict(rows=n, shapes_ok=shapes_ok, finite=fin, max_abs_err=err, rel_err=err / scale,
                         side_files=side)
    except Exception as e:  # noqa: BLE001
        return pid, dict(error=str(e))


def is_bad(v):
    if "error" in v:
        return True
    return not (v["shapes_ok"] and v["finite"] and v["side_files"] and v["rel_err"] < 2e-3)


def summarize(pids, res):
    bad = {}
    rows_total = 0
    max_rel = None
    for k, v in res.items():
        if is_bad(v):
            bad[k] = v
        rows_total += v.get("rows", 0)
        e = v.get("rel_err", 0)
        if max_rel is None or e > max_rel:
            max_rel = e
    return dict(plays_source=len(pids), plays_checked=len(res), plays_bad=len(bad), bad=bad,
                rows_total=int(rows_total), max_rel_err=0 if max_rel is None else max_rel)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    pids = []
    for f in glob.glob(f"{SRC}/*.json"):
        pids.append(os.path.basename(f)[:-5])
    pids.sort()
    os.makedirs(DST, exist_ok=True)
    print(f"{len(pids)} plays in {SRC}", flush=True)
    with Pool(a.procs) as pool:
        if not a.verify:
            for i, (pid, msg) in enumerate(pool.imap_unordered(one, pids)):
                if i % 200 == 0:
                    print(f"  {i}/{len(pids)} {pid}: {msg}", flush=True)
            return
        res = dict(pool.imap_unordered(check, pids))
    summary = summarize(pids, res)
    with open(f"{DST}/verify.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    short = {}
    for k, v in summary.items():
        if k != "bad":
            short[k] = v
    print(json.dumps(short, indent=1), "\nbad:", list(summary["bad"])[:20], flush=True)


if __name__ == "__main__":
    main()
