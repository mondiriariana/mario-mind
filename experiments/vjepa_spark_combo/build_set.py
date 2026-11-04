"""Frame set for VLA-JEPA and the V-JEPA + Spark analyses (CPU, emulator env $MJG_PY: needs stable-retro).

Every non-held-out play, up to --per-play moment-index moments each (Mario in control), replayed from the .bk2 to
get the screen the model sees, plus the V-JEPA 2.1 G row for the same moment (the 2 s clip ending at that screen).
Labels: the moment index plus Mario's screen position from RAM. Held-out fMRI sessions are never used.

Inputs: $MJG_ROOT/results/moment_index/v1/moments.csv, $MJG_ROOT/features/vjepa21G_c16s8_f8_g8/<play>.npy
(+ .last.npy, .game.npz), the .bk2 replays in $MJG_DATA and the game files in $MJG_INTEGRATION.

Writes to --out:
  frames.npz   uint8 [n, 224, 240, 3]
  labels.csv   one row per frame; `lm_subset` marks the frames used for the Spark language-model pass
               (analyze.py --exp 3)
  vjepa.npz    float16: depth_mean [n, 4, 1664] (blocks 11, 23, 37, 47, mean over the 8x8 grid),
               b47_grid2 [n, 4, 1664] (block 47 pooled to 2x2), last_mean [n, 1664] (newest tubelet, block 47)

  python build_set.py --out <frame set dir>
"""
import argparse, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd

ROOT = os.environ.get("MJG_ROOT", ".")
SCRATCH = os.environ.get("MJG_SCRATCH", os.path.join(ROOT, "scratch"))
CNEUROMOD = os.environ.get("MJG_CNEUROMOD", os.path.join(ROOT, "cneuromod"))
DATA = os.environ.get("MJG_DATA", f"{CNEUROMOD}/data/mario")
INTEG = os.environ.get("MJG_INTEGRATION", f"{CNEUROMOD}/mario.stimuli")
FEAT = f"{ROOT}/features/vjepa21G_c16s8_f8_g8"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # the mjg package


def work(args):
    """Replay one play and gather its screens and V-JEPA rows."""
    from mjg import emulator as E
    pid, bk2, screens, rows = args
    want = set()
    for s in screens:
        want.add(int(s))
    rep = E.replay_bk2(bk2, INTEG, keep_frame=lambda k: k in want)
    fr = {}
    for k, f in zip(rep["frame_idx"], rep["frames"]):
        fr[int(k)] = f
    z = np.load(f"{FEAT}/{pid}.npy", mmap_mode="r")
    zl = np.load(f"{FEAT}/{pid}.last.npy", mmap_mode="r")
    dm, g2, lm = [], [], []
    for r in rows:
        x = np.asarray(z[int(r)], dtype=np.float32)                      # [4, 64, 1664]
        dm.append(x.mean(1))
        b = x[3].reshape(8, 8, -1)
        g2.append(b.reshape(2, 4, 2, 4, -1).mean((1, 3)).reshape(4, -1))  # 2x2 quadrants
        lm.append(np.asarray(zl[int(r)], dtype=np.float32).mean(0))
    shots = []
    for s in screens:
        shots.append(fr[int(s)])
    return pid, shots, np.stack(dm), np.stack(g2), np.stack(lm)


def stacked(res, i):
    """Field i of every play's result, concatenated in play order, as float16."""
    parts = []
    for r in res:
        parts.append(r[i])
    return np.concatenate(parts).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{SCRATCH}/vjepa_spark_combo/v1")
    ap.add_argument("--per-play", type=int, default=12)
    ap.add_argument("--lm-n", type=int, default=2400, help="frames for the Spark language-model pass")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    from mjg import data as D
    mi = pd.read_csv(f"{ROOT}/results/moment_index/v1/moments.csv")
    mi = mi[~mi["held_out"]]
    bk2 = {}
    for p in D.index_plays(DATA):
        bk2[p.play_id] = p.bk2
    jobs, rows = [], []
    for pid, m in mi.groupby("play_id", sort=True):
        m = m.sample(min(a.per_play, len(m)), random_state=1).sort_values("screen")
        g = np.load(f"{FEAT}/{pid}.game.npz")
        for _, r in m.iterrows():
            s = int(r["screen"])
            d = r.to_dict()
            d["x_screen"] = int(g["ram_player_x_posHi"][s] * 256 + g["ram_player_x_posLo"][s]
                                - (g["ram_xscrollHi"][s] * 256 + g["ram_xscrollLo"][s]))
            d["y_pos"] = int(g["ram_player_y_pos"][s])
            rows.append(d)
        jobs.append((pid, bk2[pid], m["screen"].values, m["vjepa_row"].values))
    print(f"{len(jobs)} plays, {len(rows)} moments", flush=True)
    with Pool(a.procs) as pool:
        res = pool.map(work, jobs, chunksize=4)
    frames = []
    for r in res:
        frames.extend(r[1])
    lab = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    lab["lm_subset"] = False
    lab.loc[rng.choice(len(lab), min(a.lm_n, len(lab)), replace=False), "lm_subset"] = True
    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/frames.npz", frames=np.stack(frames))
    np.savez(f"{a.out}/vjepa.npz", depth_mean=stacked(res, 2), b47_grid2=stacked(res, 3), last_mean=stacked(res, 4))
    lab.to_csv(f"{a.out}/labels.csv", index=False)
    print(len(frames), "frames |", lab["sub"].value_counts().to_dict(), "| sessions",
          lab.groupby(["sub", "ses"]).ngroups, "| classes", lab["human_class"].value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
