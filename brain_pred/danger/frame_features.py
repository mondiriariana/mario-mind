"""Danger experiment, step 1 (CPU): what is around Mario at every moment of every human play, read from game memory.

Per play (features/ram_full/<play>.npz, RAM after each frame), every --every frames (default 6 = 10 per second):
  mario    x on screen (0-1), height (0-1, 1 = bottom), x / y speed, airborne, big
  enemies  number on screen; nearest enemy ahead (dx in px, dy), nearest behind (dx)
  holes    distance to the nearest gap in the floor ahead (px, in the level map in memory; 999 = none in 4 screens
           tiles), width of that gap (tiles)
  state    dying (player state 0x06 / 0x0B), fallen (below the screen), lives
and every death: frame and cause ("fall" = below the screen, "enemy" = other dying states; timer deaths are rare and
counted as "enemy").
Writes results/danger/frames/<play>.npz. Resumable.

Memory map (SMB, as in experiments/glimmer_imitation/rl.py and brain_pred/encoding/build_tr_features.py): Mario x =
0x6D*256 + 0x86, y = 0xCE; enemies drawn 0x0F+i, x = 0x6E+i*256 + 0x87+i, y = 0xCF+i; player state 0x0E; vertical
screen 0xB5; lives 0x75A; level map 0x500-0x69F = 2 pages x 13 rows x 16 columns of metatiles (0 = empty), page =
(x // 256) % 2, row r covers y = 32 + 16 r; the floor is rows 11-12.

  $MJG_SCI_PY frame_features.py --subjects sub-01 sub-02 sub-03 sub-05 sub-06 --procs 32    (CPU node)
"""
import argparse, glob, os, time
from multiprocessing import Pool

import numpy as np

ROOT = os.environ.get("MJG_ROOT", ".")
RAM = f"{ROOT}/features/ram_full"
OUT = f"{ROOT}/results/danger/frames"


def s8(v):
    v = np.asarray(v).astype(np.int32)
    return np.where(v > 127, v - 256, v)


def gap_ahead(ram_row, mx, max_tiles=24):
    """Distance (px) from Mario to the first floor column ahead with rows 11 and 12 empty, and its width (tiles)."""
    t0 = mx // 16
    for k in range(1, max_tiles + 1):
        col = t0 + k
        page, c = (col // 16) % 2, col % 16
        base = 0x500 + page * 0xD0
        if ram_row[base + 11 * 16 + c] == 0 and ram_row[base + 12 * 16 + c] == 0:
            w = 1
            while w < 8:
                col2 = col + w
                p2, c2 = (col2 // 16) % 2, col2 % 16
                b2 = 0x500 + p2 * 0xD0
                if not (ram_row[b2 + 11 * 16 + c2] == 0 and ram_row[b2 + 12 * 16 + c2] == 0):
                    break
                w += 1
            return col * 16 - mx, w
    return 999, 0


def features(ram, every):
    n = len(ram)
    idx = np.arange(0, n, every)
    r = ram[idx].astype(np.int32)
    mx = r[:, 0x6D] * 256 + r[:, 0x86]
    scroll = r[:, 0x71A] * 256 + r[:, 0x71C]
    my = r[:, 0xCE].astype(np.float32)
    drawn, ex, ey = [], [], []
    for i in range(5):  # the 5 enemy slots
        drawn.append(r[:, 0x0F + i] != 0)
        ex.append(r[:, 0x6E + i] * 256 + r[:, 0x87 + i])
        ey.append(r[:, 0xCF + i])
    drawn, ex, ey = np.stack(drawn, 1), np.stack(ex, 1), np.stack(ey, 1).astype(np.float32)
    dx = ex - mx[:, None]
    ahead = drawn & (dx >= 0)
    behind = drawn & (dx < 0)
    ea = np.where(ahead, dx, 9999).min(1)
    eb = np.where(behind, -dx, 9999).min(1)
    j = np.where(ahead, dx, 9999).argmin(1)
    ea_dy = np.where(ahead.any(1), ey[np.arange(len(j)), j] - my, 0)
    gaps = []
    for fi, x in zip(idx, mx):
        gaps.append(gap_ahead(ram[fi], int(x)))
    gaps = np.array(gaps)
    return dict(frame=idx, mx_screen=((mx - scroll) / 256).astype(np.float32), my=(my / 240).astype(np.float32),
                vx=(s8(r[:, 0x57]) / 40).astype(np.float32), vy=(s8(r[:, 0x9F]) / 8).astype(np.float32),
                air=(r[:, 0x1D] != 0).astype(np.float32), big=(r[:, 0x754] == 0).astype(np.float32),
                n_enemies=drawn.sum(1).astype(np.float32),
                enemy_ahead_dx=np.minimum(ea, 999).astype(np.float32), enemy_ahead_dy=(ea_dy / 240).astype(np.float32),
                enemy_behind_dx=np.minimum(eb, 999).astype(np.float32),
                gap_ahead_dx=gaps[:, 0].astype(np.float32), gap_width=gaps[:, 1].astype(np.float32),
                dying=np.isin(r[:, 0x0E], (6, 11)).astype(np.float32), fallen=(r[:, 0xB5] > 1).astype(np.float32),
                lives=r[:, 0x75A].astype(np.float32))


def deaths(ram):
    """(frame, cause) of each death: first frame of a dying state or of falling below the screen."""
    st = np.isin(ram[:, 0x0E], (6, 11))
    fall = ram[:, 0xB5] > 1
    bad = st | fall
    on = np.where(bad & ~np.r_[False, bad[:-1]])[0]
    out = []
    for f in on:
        if out and f - out[-1][0] < 120:  # one death = one episode of the dying state
            continue
        win = fall[f:f + 60]
        out.append((int(f), "fall" if win.any() and not st[f] else "enemy"))
    return out


def one(job):
    pid, every = job
    dst = f"{OUT}/{pid}.npz"
    if os.path.exists(dst):
        return pid, "exists"
    ram = np.load(f"{RAM}/{pid}.npz")["ram"]
    f = features(ram, every)
    frames, causes = [], []
    for frame, cause in deaths(ram):
        frames.append(frame)
        causes.append(cause)
    np.savez_compressed(dst + ".tmp.npz", **f, death_frame=np.array(frames, int),
                        death_cause=np.array(causes, dtype="U5"), n_frames=len(ram))
    os.replace(dst + ".tmp.npz", dst)
    return pid, f"{len(ram)} frames, deaths {causes}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", default=["sub-03"])
    ap.add_argument("--plays", nargs="*", default=None, help="only these play ids")
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pids = []
    for s in a.subjects:
        for p in glob.glob(f"{RAM}/{s}_*.npz"):
            pids.append(os.path.basename(p)[:-4])
    pids.sort()
    if a.plays:
        wanted = set(a.plays)
        kept = []
        for p in pids:
            if p in wanted:
                kept.append(p)
        pids = kept
    jobs = []
    for p in pids:
        jobs.append((p, a.every))
    t0 = time.time()
    with Pool(a.procs) as pool:
        for i, (pid, msg) in enumerate(pool.imap_unordered(one, jobs)):
            if i % 100 == 0 or a.plays:
                print(f"[{i + 1}/{len(pids)}] {pid}: {msg} ({time.time() - t0:.0f}s)", flush=True)
    print(f"done: {len(pids)} plays ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
