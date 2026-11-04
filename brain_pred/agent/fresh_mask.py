"""Fresh-state masks (CPU): keep only the scans where the agent's state is at most one scan old.

The agent's scan states (embed_scans.py) come from turns cut by build_scan_turns.py, which stops at Mario's first
death in a play, but plays go on through later lives: the scans after the last turn would hold a frozen pre-death
state. V-JEPA is computed per scan.

This writes, for each run, a new agent tag <src><suffix> whose states are the same file (symlink) and whose mask keeps
only the scans where the state is fresh (the decision started <= --max-age s before the scan ends: in practice
Mario's first life). encode.py --agent-tags <src><suffix> then fits and scores every model on those scans only.

  python fresh_mask.py --subject sub-03 --src sft plain      (CPU, $MJG_SCI_PY)
"""
import argparse, os, re, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mjg import encoding as E  # noqa: E402

ROOT = os.environ.get("MJG_ROOT", ".")
SCRATCH = os.environ.get("MJG_SCRATCH", os.path.join(ROOT, "scratch"))
TRF = f"{ROOT}/results/brain_encoding/tr_features"
TURNS = f"{SCRATCH}/glimmer_scan"


def play_id(stem):  # as brain_pred/agent/embed_scans.py
    m = re.match(r"(sub-\d+)_(ses-\d+)_task-mario_level-(w\dl\d)_rep-(\d+)", stem)
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}_rep-{m.group(4)}" if m else stem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="sub-03")
    ap.add_argument("--src", nargs="+", default=["sft", "plain"])
    ap.add_argument("--suffix", default="fresh")
    ap.add_argument("--max-age", type=float, default=1.49)
    ap.add_argument("--turns-split", default="all",
                    help="the turns the states were computed on (all_lives: every life)")
    a = ap.parse_args()
    sub = a.subject
    n_trs = E.run_trs_from_confounds(sub)
    st = E.scan_table(sub, n_trs)
    cache, tot, kept = {}, 0, 0
    for key in sorted(n_trs):
        g = st[st.key == key].sort_values("tr")
        srcs = []
        for s in a.src:
            srcs.append(f"{TRF}/{sub}/{key}.agent_{s}.npy")
        all_there = True
        for p in srcs:
            if not os.path.exists(p):
                all_there = False
                break
        if not all_there:
            continue
        m0 = np.load(srcs[0].replace(".npy", ".mask.npy"))
        fresh = np.zeros(len(m0), bool)
        for r in g.itertuples():
            t = int(r.tr)
            if t >= len(m0) or not r.stem or r.f1 <= 0:
                continue
            pid = play_id(r.stem)
            if pid not in cache:
                p = f"{TURNS}/{sub}/{a.turns_split}/{pid}.npz"
                cache[pid] = np.load(p)["t"] if os.path.exists(p) else None
            tt = cache[pid]
            if tt is None or not len(tt):
                continue
            i = int(np.searchsorted(tt, r.f1, side="left")) - 1  # the decision embed_scans.py used
            if i >= 0 and (r.f1 - tt[i]) / 60.0 <= a.max_age:
                fresh[t] = True
        for s, p in zip(a.src, srcs):
            m = np.load(p.replace(".npy", ".mask.npy")) & fresh
            dst = f"{TRF}/{sub}/{key}.agent_{s}{a.suffix}.npy"
            if not os.path.lexists(dst):
                os.symlink(os.path.basename(p), dst)
            np.save(dst.replace(".npy", ".mask.npy"), m)
        tot += int(m0.sum())
        kept += int((m0 & fresh).sum())
    tags = []
    for s in a.src:
        tags.append(s + a.suffix)
    print(f"{sub}: kept {kept} of {tot} scans with an agent state ({kept / max(1, tot):.1%}) as fresh "
          f"(<= {a.max_age} s); tags {tags}")


if __name__ == "__main__":
    main()
