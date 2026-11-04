"""How Glimmer agents play (e.g. plain vs SFT), next to the human, with uncertainty.

Reads glimmer_play.py outputs (<dir>/<level>/run-NNN/{meta.json, steps.jsonl}). The human reference is the
validation split of the SFT data (build_sft.py; held-out sessions, never trained on), restricted to the same levels.
Per agent, over episodes (95% stratified-by-level bootstrap CIs, 2,000 resamples; Agarwal et al. 2021):
  died        share of episodes ending in death          turns     turns survived (cap = --max-turns)
  progress    pixels gained                               valid     share of turns with a parseable answer
  jump        share of turns with a jump combo            run_right share of turns holding RIGHT+B / RIGHT+A+B
  KL          KL(human || agent) of the 14-combo action mix (pooled turns; primary human-likeness number)
Reading: compare the agents' KL intervals first, then deaths and progress; overlapping intervals = no detectable
difference at this sample size.

  python compare_play.py --runs plain=<dir> sft=<dir> --human <build_sft.py out> [--out table.json]
"""
import argparse
import glob
import json
import os

import numpy as np

ROOT = os.environ.get("MJG_ROOT", ".")
SCRATCH = os.environ.get("MJG_SCRATCH", os.path.join(ROOT, "scratch"))
# 14-combo indices (mjg.actions.COMBOS: NOOP, RIGHT, RIGHT+A, RIGHT+B, RIGHT+A+B, LEFT, LEFT+A, LEFT+B, LEFT+A+B,
# A, B, A+B, DOWN, UP)
JUMP = [2, 4, 6, 8, 9, 11]
RUN_R = [3, 4]


def dist(combos):
    c = np.bincount(np.asarray(combos, int), minlength=14).astype(float) + 0.5  # add-half smoothing
    return c / c.sum()


def kl(p, q):
    return float((p * np.log(p / q)).sum())


def episodes(d):
    out = []
    for m in sorted(glob.glob(f"{d}/*/run-*/meta.json")):
        meta = json.load(open(m))
        st = []
        for line in open(m.replace("meta.json", "steps.jsonl")):
            st.append(json.loads(line))
        c, ok = [], []
        for s in st:
            c.append(s.get("combo", 0))
            ok.append(s.get("error") is None)
        out.append(dict(level=meta["level"], died=float(meta["end"] == "death"), turns=meta["turns"],
                        progress=meta["progress"], valid=np.mean(ok) if st else 0, combos=c,
                        jump=np.isin(c, JUMP).mean() if c else 0, run_right=np.isin(c, RUN_R).mean() if c else 0))
    return out


def boot(eps, stat, ph, n=2000, seed=0):
    """Point value and 95% CI of stat(episodes), resampling episodes within each level."""
    rng = np.random.default_rng(seed)
    by = {}
    for e in eps:
        by.setdefault(e["level"], []).append(e)
    vals = []
    for _ in range(n):
        smp = []
        for lv in by.values():
            for i in rng.integers(0, len(lv), len(lv)):
                smp.append(lv[i])
        vals.append(stat(smp, ph))
    return stat(eps, ph), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def mean_of(key):
    """Stat: mean of an episode field."""
    def stat(eps, ph):
        vals = []
        for x in eps:
            vals.append(x[key])
        return np.mean(vals)
    return stat


def share_of(group):
    """Stat: share of the pooled turns whose combo is in group."""
    def stat(eps, ph):
        hits = []
        for x in eps:
            hits.append(np.isin(x["combos"], group))
        return np.mean(np.concatenate(hits))
    return stat


def kl_stat(eps, ph):
    combos = []
    for x in eps:
        combos.append(x["combos"])
    return kl(ph, dist(np.concatenate(combos)))


STATS = {
    "died": mean_of("died"),
    "turns": mean_of("turns"),
    "progress": mean_of("progress"),
    "valid": mean_of("valid"),
    "jump": share_of(JUMP),
    "run_right": share_of(RUN_R),
    "KL": kl_stat,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="name=dir")
    ap.add_argument("--human", default=f"{SCRATCH}/glimmer_sft/v1")
    ap.add_argument("--out", default=None, help="write the table as json")
    a = ap.parse_args()
    agents = {}
    for spec in a.runs:
        name, d = spec.split("=", 1)
        agents[name] = episodes(d)
    level_set = set()
    for eps in agents.values():
        for e in eps:
            level_set.add(e["level"])
    levels = sorted(level_set)
    hc = []
    for r in map(json.loads, open(f"{a.human}/index.jsonl")):
        if r.get("n_turns") and r["split"] == "val" and r["level"] in levels:
            hc += list(np.load(f"{a.human}/val/{r['play_id']}.npz")["combo"])
    ph = dist(hc)
    hj, hr = np.isin(hc, JUMP).mean(), np.isin(hc, RUN_R).mean()
    print(f"levels {levels}; human reference: {len(hc)} held-out turns (jump {hj:.2f}, run-right {hr:.2f})")
    res = {}
    for name, eps in agents.items():
        res[name] = {}
        for k, f in STATS.items():
            res[name][k] = boot(eps, f, ph)
        res[name]["episodes"] = len(eps)
        print(f"\n{name}: {len(eps)} episodes")
        for k in STATS:
            v, lo, hi = res[name][k]
            print(f"  {k:10s} {v:8.3f}   [{lo:.3f}, {hi:.3f}]")
    if a.out:
        table = {}
        for name, r in res.items():
            table[name] = {}
            for k, v in r.items():
                table[name][k] = list(v) if isinstance(v, tuple) else v
        json.dump(dict(levels=levels, human_turns=len(hc), human_jump=hj, human_run_right=hr, agents=table),
                  open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
