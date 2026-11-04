"""Score every iteration of RL runs from their saved games, with the game-phase-matched human comparison (CPU).

rl.py's rollouts are --turns turns from a level start, so they are compared with the humans' turns from the same
phase (the first --turns turns of each validation play); over whole levels, a classifier mostly measures where in
the level Mario is, not how he plays. Per run and iteration, from <run>/rollouts/*/run-*/{steps.jsonl, ram.npy}:
  human_likeness_matched  cross-validated agent-vs-human logistic accuracy on x (0.5 = indistinguishable)
  action_kl_matched       KL(human || agent) of the button-combo mix
  turns, deaths
  $MJG_MUSE_PY rl_eval.py --runs density=<dir> gail=<dir> airl=<dir> --out rl_eval.json
"""
import argparse, glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl  # noqa: E402


def agent_turns(run_dir):
    """-> {iteration: (X [n, d], deaths)} from the saved games."""
    out = {}
    for d in sorted(glob.glob(f"{run_dir}/rollouts/*/run-*")):
        if not os.path.exists(f"{d}/ram.npy"):
            continue
        ram = np.load(f"{d}/ram.npy")
        prev = 0
        for i, line in enumerate(open(f"{d}/steps.jsonl")):
            s = json.loads(line)
            if i >= len(ram) or "it" not in s:
                break
            c = int(s.get("combo") or 0)
            X, n_dead = out.setdefault(int(s["it"]), ([], [0]))
            X.append(rl.x_of(rl.ram_features(ram[i]), prev, c))
            n_dead[0] += int(bool(s.get("done")) and s.get("reason") == "death")
            prev = c
    res = {}
    for it, (X, dead) in sorted(out.items()):
        res[it] = (np.stack(X), dead[0])
    return res


def combo_mix(X):
    """Smoothed share of each button combo (the last N_ACT columns of x are the combo, one-hot)."""
    h = np.bincount(X[:, -rl.N_ACT:].argmax(1), minlength=rl.N_ACT) + 1
    return h / h.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="name=run dir (results/glimmer_imitation/rl/<tag>)")
    ap.add_argument("--turns", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    Xv, _, _, Tv = rl.human_turns("val", with_turn=True)
    Xm = Xv[Tv < a.turns]
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-6
    nz = lambda X: (X - mu) / sd
    hm = combo_mix(Xm)
    print(f"human validation turns {len(Xv)}, of which {len(Xm)} in the first {a.turns} turns of a play")
    res = {}
    for run in a.runs:
        name, d = run.split("=", 1)
        res[name] = []
        for it, (X, dead) in agent_turns(d).items():
            pa = combo_mix(X)
            row = dict(it=it, turns=len(X), deaths=dead, human_likeness_matched=rl.human_likeness(nz(X), nz(Xm)),
                       action_kl_matched=float((hm * np.log(hm / pa)).sum()))
            res[name].append(row)
            print(f"{name:8s} it {it:2d}  turns {row['turns']:3d}  deaths {dead:2d}  matched clf acc "
                  f"{row['human_likeness_matched']:.3f}  matched action KL {row['action_kl_matched']:.3f}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
