"""Danger experiment, step 3a (CPU): pick scenario start moments for the agent, from human plays.

A scenario = a human play replayed up to a frame, then the agent takes over (glimmer_play --scenarios). The moment
the agent faces is chosen by threat type, from the risk files (risk_model.py):
  enemy  an enemy ahead (within the risk model's "near" distance) at medium-high risk (>= --risk-min)
  hole   a gap ahead at medium-high risk
  calm   no threat and risk < --calm-max for the next 5 s
The agent takes over --lead seconds before that moment (so it plays into the situation), at least --history seconds
into the play (the brain encoder needs the previous scans), and not within 10 s after a death. Moments come from the
non-held-out sessions only, at most --per-level per level and type, spread over subjects and levels.
Writes results/danger/scenarios_<tag>.json: [{name, type, level, bk2, frame, risk, play}].

  $MJG_SCI_PY scenarios.py --n 15 --tag v1      (CPU)
"""
import argparse, glob, json, os, re

import numpy as np

ROOT = os.environ.get("MJG_ROOT", ".")
CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = f"{ROOT}/results/danger"
RAM = f"{ROOT}/features/ram_full"
KINDS = ("enemy", "hole", "calm")


def calm_moments(risk, typ, calm_max):
    """True where the next 50 samples (5 s) have no threat and risk < calm_max."""
    ok = []
    for i in range(len(risk)):
        if i + 50 <= len(risk):
            ok.append(np.nanmax(risk[i:i + 50]) < calm_max and (typ[i:i + 50] == "none").all())
        else:
            ok.append(False)
    return np.array(ok)


def all_full(picked, n):
    for v in picked.values():
        if len(v) < n:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=15, help="scenarios per type")
    ap.add_argument("--risk-min", type=float, default=0.19)
    ap.add_argument("--calm-max", type=float, default=0.05)
    ap.add_argument("--lead", type=float, default=1.5)
    ap.add_argument("--history", type=float, default=15.0)
    ap.add_argument("--per-level", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    hold = json.load(open(os.path.join(CODE, "configs", "holdout_sessions.json")))["holdout"]
    rng = np.random.default_rng(a.seed)
    pids = []
    for p in glob.glob(f"{OUT}/risk/*.npz"):
        pids.append(os.path.basename(p)[:-4])
    pids.sort()
    rng.shuffle(pids)
    lead, hist = int(a.lead * 60), int(a.history * 60)
    picked = {}
    for kind in KINDS:
        picked[kind] = []
    per_level = {}
    for pid in pids:
        m = re.match(r"(sub-\d+)_(ses-\d+)_(w\dl\d)_", pid)
        if m.group(2) in set(hold.get(m.group(1), [])):
            continue
        z = np.load(f"{OUT}/risk/{pid}.npz")
        f, risk, typ, deaths = z["frame"], z["risk"], z["type"], z["death_frame"]
        meta = json.load(open(f"{RAM}/{pid}.json"))
        for kind in KINDS:
            if len(picked[kind]) >= a.n or per_level.get((kind, m.group(3)), 0) >= a.per_level:
                continue
            if kind == "calm":
                ok = calm_moments(risk, typ, a.calm_max)
            else:
                ok = (typ == kind) & (risk >= a.risk_min)
            ok &= (f - lead >= hist)
            for d in deaths:  # not right after a death, and the human does not die before the agent takes over
                ok &= ~((f - lead >= d) & (f - lead < d + 600)) & ~((d >= f - lead) & (d < f))
            idx = np.where(ok)[0]
            if not len(idx):
                continue
            i = int(rng.choice(idx))
            start = int(f[i] - lead)
            picked[kind].append(dict(name=f"{kind}_{len(picked[kind]):02d}_{pid}", type=kind, level=m.group(3),
                                     bk2=meta["bk2"], frame=start, target_frame=int(f[i]),
                                     risk=float(risk[i]), play=pid))
            per_level[(kind, m.group(3))] = per_level.get((kind, m.group(3)), 0) + 1
        if all_full(picked, a.n):
            break
    out = []
    for kind in KINDS:
        out.extend(picked[kind])
    json.dump(out, open(f"{OUT}/scenarios_{a.tag}.json", "w"), indent=1)
    for k, v in picked.items():
        levels, risks, subjects = set(), [], set()
        for scen in v:
            levels.add(scen["level"])
            risks.append(scen["risk"])
            subjects.add(scen["play"][:6])
        print(f"{k}: {len(v)} scenarios, levels {sorted(levels)}, "
              f"mean risk {np.mean(risks):.3f}, subjects {sorted(subjects)}")
    print("wrote", f"{OUT}/scenarios_{a.tag}.json")


if __name__ == "__main__":
    main()
