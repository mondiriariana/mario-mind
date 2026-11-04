"""Calibration data, part 1 (CPU): every play of one subject's fMRI sessions, cut into the agent's turns.

Same cutting as the agent's imitation data (experiments/glimmer_imitation/build_sft.py: replay the .bk2, turns from
the first frame Mario is in control, Odysseus holds, observation = every 10th frame of the previous hold + the
last; stops at the first death), but for ALL sessions, the held-out fMRI sessions included: the encoder is tested on
those, so the agent's state must exist there too (the agent never trains on these files).
Writes $MJG_SCRATCH/glimmer_scan/<sub>/all/<play_id>.npz (frames, obs_start, obs_len, combo, t per turn).

  python build_scan_turns.py --subject sub-03      (env with stable-retro: $MJG_PY)
"""
import argparse, os, sys, time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments", "glimmer_imitation"))
from build_sft import work, DATA  # noqa: E402  (same cutting as the imitation data)

ROOT = os.environ.get("MJG_ROOT", ".")
SCRATCH = os.environ.get("MJG_SCRATCH", os.path.join(ROOT, "scratch"))
OUT = f"{SCRATCH}/glimmer_scan"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="sub-03")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--all-lives", action="store_true",
                    help="turns through every life of a play (not only until the first death), written to "
                         "<sub>/all_lives/ with a per-turn life index; for embed_scans.py --turns-split all_lives")
    a = ap.parse_args()
    from mjg.data import index_plays
    plays = index_plays(DATA, [a.subject])
    out = f"{OUT}/{a.subject}"
    split = "all_lives" if a.all_lives else "all"
    jobs = []
    for p in plays:
        jobs.append((dict(vars(p)), split, out, a.every, a.all_lives))  # Play is a dataclass
    t0, n = time.time(), 0
    with Pool(a.procs) as pool:
        for _ in pool.imap_unordered(work, jobs):
            n += 1
            if n % 50 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} plays ({time.time() - t0:.0f}s)", flush=True)
    print(f"{a.subject}: {len(jobs)} plays -> {out}/{split}")


if __name__ == "__main__":
    main()
