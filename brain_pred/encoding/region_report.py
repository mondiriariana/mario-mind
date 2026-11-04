"""Where in the brain does each encoder predict well? Per-network and per-region breakdown of held-out accuracy (CPU).

Whole-brain means stay low because most regions are not driven by the game; papers report where a model works.
Reads encode.py's per-model results ($MJG_ROOT/results/brain_encoding/<sub>[_lowlevel]/<model>.npz: held-out r and
gain over the nuisance model per Schaefer-1000 parcel) and the atlas labels (7 networks). Reports, per model:
  per network: mean r, mean gain, share of parcels with gain > 0, best parcel
  well-predicted regions: parcels with held-out r >= --r-min (and their networks)
  mean gain inside target parcel sets: parcels the nuisance model predicts with r >= --r-min, and parcels whose
  human BOLD rises with danger (--danger-npz)
and, per pair of models (--compare a,b): parcels where a beats b by more than --margin, by network.
Writes <out>.json and prints markdown tables.

  python region_report.py --subject sub-03 --suffix _lowlevel --models nuis,gamestate,vjepa4,agent_sftfresh_d1 \
      --compare agent_sftfresh_d1,vjepa4 --out results/brain_encoding/region_report_sub-03.json
"""
import argparse, json, os

import numpy as np

ROOT = os.environ.get("MJG_ROOT", ".")
CNEUROMOD = os.environ.get("MJG_CNEUROMOD", os.path.join(ROOT, "cneuromod"))
ATLAS = os.environ.get("MJG_ATLAS", os.path.join(CNEUROMOD, "atlas"))
LABELS = f"{ATLAS}/tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-1000Parcels7Networks_dseg.tsv"
NETS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NET_NAME = dict(Vis="visual", SomMot="somatomotor", DorsAttn="dorsal attention",
                SalVentAttn="salience / ventral attention", Limbic="limbic", Cont="control (frontoparietal)",
                Default="default mode")


def labels():
    """-> parcel names (column 0..999 = parcel 1..1000) and the network of each."""
    rows = []
    for line in open(LABELS):
        rows.append(line.rstrip("\n").split("\t"))
    names = []
    for r in sorted(rows[1:], key=lambda r: int(r[0])):
        names.append(r[1])
    nets = []
    for n in names:
        nets.append(n.split("_")[2])
    return names, np.array(nets)


def per_network(f):
    """f(network) for each of the 7 networks -> {network: value}."""
    out = {}
    for n in NETS:
        out[n] = f(n)
    return out


def network_cells(f):
    """f(network) -> str for each of the 7 networks, as a list."""
    cells = []
    for n in NETS:
        cells.append(f(n))
    return cells


def network_row(f):
    """Markdown cells, one per network."""
    return " | ".join(network_cells(f))


def top_parcels(names, r, g):
    out = []
    for i in np.argsort(-r)[:10]:
        out.append(dict(parcel=names[i], r=float(r[i]), gain=float(g[i])))
    return out


def model_report(names, net, r, g, r_min):
    per = {}
    for n in NETS:
        k = net == n
        per[n] = dict(n=int(k.sum()), mean_r=float(r[k].mean()), mean_gain=float(g[k].mean()),
                      share_gain_pos=float((g[k] > 0).mean()), best_parcel=names[int(np.where(k)[0][np.argmax(r[k])])],
                      best_r=float(r[k].max()))
    good = r >= r_min
    return dict(per_network=per, mean_r=float(r.mean()), n_r_above=int(good.sum()),
                r_above_by_network=per_network(lambda n: int((good & (net == n)).sum())),
                top10=top_parcels(names, r, g))


def compare_report(names, net, res, x, y, margin):
    diff = res[x]["r"] - res[y]["r"]
    win = diff > margin
    top_wins = []
    for i in np.argsort(-diff)[:10]:
        top_wins.append(dict(parcel=names[i], diff=float(diff[i]), r_a=float(res[x]["r"][i]),
                             r_b=float(res[y]["r"][i])))
    return dict(n_win=int(win.sum()), by_network=per_network(lambda n: int((win & (net == n)).sum())),
                mean_diff_by_network=per_network(lambda n: float(diff[net == n].mean())), top_wins=top_wins)


def target_sets(a, res):
    """Parcel sets shared by every model: reliably predicted by the nuisance model alone, and the parcels whose
    human BOLD rises with danger (brain_pred/danger/human_response.py, t high - safe > --danger-t)."""
    sets = {}
    if "nuis" in res:
        sets[f"nuisance r >= {a.r_min}"] = res["nuis"]["r"] >= a.r_min
    if a.danger_npz and os.path.exists(a.danger_npz):
        t = np.load(a.danger_npz)["t_high_minus_safe"]
        sets[f"human danger t > {a.danger_t}"] = t > a.danger_t
    return sets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="sub-03")
    ap.add_argument("--suffix", default="_lowlevel",
                    help="results folder suffix (encode.py --lowlevel writes _lowlevel)")
    ap.add_argument("--models", required=True)
    ap.add_argument("--compare", nargs="*", default=[], help="a,b pairs: where a beats b")
    ap.add_argument("--r-min", type=float, default=0.3)
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--danger-npz", default=f"{ROOT}/results/danger/human_sub-03_parcels.npz")
    ap.add_argument("--danger-t", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    names, net = labels()
    d = f"{ROOT}/results/brain_encoding/{a.subject}{a.suffix}"
    res = {}
    for m in a.models.split(","):
        res[m] = dict(np.load(f"{d}/{m}.npz"))
    out = dict(subject=a.subject, suffix=a.suffix, r_min=a.r_min, models={}, compare={})
    print(f"## {a.subject}: per network (mean held-out r / mean gain over nuisance / share of parcels with gain > 0)\n")
    print("| model | " + network_row(lambda n: NET_NAME[n]) + " | whole brain |")
    print("|---" * (len(NETS) + 2) + "|")
    for m, z in res.items():
        r, g = z["r"], z["gain"]
        o = model_report(names, net, r, g, a.r_min)
        out["models"][m] = o
        cells = network_row(lambda n: f"{o['per_network'][n]['mean_r']:.3f} / {o['per_network'][n]['mean_gain']:+.3f}"
                                      f" / {o['per_network'][n]['share_gain_pos']:.0%}")
        print(f"| {m} | " + cells + f" | {r.mean():.3f} / {g.mean():+.3f} / {(g > 0).mean():.0%} |")
    print(f"\n## well-predicted regions (held-out r >= {a.r_min}), count by network\n")
    print("| model | total | " + network_row(lambda n: NET_NAME[n]) + " |")
    print("|---" * (len(NETS) + 2) + "|")
    for m, o in out["models"].items():
        print(f"| {m} | {o['n_r_above']} | " + network_row(lambda n: str(o["r_above_by_network"][n])) + " |")
    sets = target_sets(a, res)
    if sets:
        print("\n## mean gain inside target parcel sets (same parcels for every model)\n")
        heads = []
        for k, v in sets.items():
            heads.append(f"{k} ({int(v.sum())})")
        print("| model | " + " | ".join(heads) + " | whole brain |")
        print("|---" * (len(sets) + 2) + "|")
        for m, z in res.items():
            gains, cells = {}, []
            for k, v in sets.items():
                gains[k] = float(z["gain"][v].mean())
                cells.append(f"{z['gain'][v].mean():+.4f}")
            out["models"][m]["target_sets"] = gains
            print(f"| {m} | " + " | ".join(cells) + f" | {z['gain'].mean():+.4f} |")
    for pair in a.compare:
        x, y = pair.split(",")
        c = compare_report(names, net, res, x, y, a.margin)
        out["compare"][pair] = c
        print(f"\n## {x} beats {y} by > {a.margin} in {c['n_win']} of 1000 parcels; by network: "
              + ", ".join(network_cells(lambda n: f"{NET_NAME[n]} {c['by_network'][n]}")))
        print("  mean r difference by network: "
              + ", ".join(network_cells(lambda n: f"{NET_NAME[n]} {c['mean_diff_by_network'][n]:+.4f}")))
        wins = []
        for t in c["top_wins"][:5]:
            wins.append(f"{t['parcel']} {t['diff']:+.3f} ({t['r_a']:.2f} vs {t['r_b']:.2f})")
        print("  largest wins: " + "; ".join(wins))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
