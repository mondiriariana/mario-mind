"""Danger experiment, step 3 (CPU): how the human brain responds to danger, on the held-out fMRI sessions.

For one subject's held-out sessions (configs/holdout_sessions.json: never used to fit anything), every scan fully
inside a play gets the danger of its moment = the highest risk (risk_model.py: chance of dying in the next 3 s) in its
1.49 s, and the threat type at that peak (enemy / hole). Scans followed by a death within 5 scans are left out, so
the response is to danger (close calls), not to dying (death animation, restart).
Brain signal: the Schaefer-1000 parcel time series minus what the nuisance encoder predicts from buttons, level, time
in play, luminance, optical flow and audio (results/bold_generator/<sub>/nuis.npz, fit on the training sessions), so
the response is beyond hands, brightness, motion and sound. Response of scan t = mean residual at t+3 and t+4
(4.5-7.5 s later: the hemodynamic delay).
Danger levels: risk quantiles of the included scans: safe (< 50th percentile), low (50-75), medium (75-90), high (> 90).
Per network (Yeo 7): mean response per level (mean over runs, SE across runs), high - safe with a permutation p-value
(risk labels shuffled within run, --perm times); per parcel: high - safe (t). Threat type: enemy vs hole at medium +
high.
Writes results/danger/human_<sub>.json and human_<sub>_parcels.npz.

  $MJG_SCI_PY human_response.py --subject sub-03     (CPU node)
"""
import argparse, json, os, re, sys

import numpy as np

ROOT = os.environ.get("MJG_ROOT", ".")
CNEUROMOD = os.environ.get("MJG_CNEUROMOD", os.path.join(ROOT, "cneuromod"))
ATLAS = os.environ.get("MJG_ATLAS", os.path.join(CNEUROMOD, "atlas"))
CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, CODE)
from mjg import encoding as E  # noqa: E402

TRF = f"{ROOT}/results/brain_encoding/tr_features"
OUT = f"{ROOT}/results/danger"
LABELS = os.path.join(ATLAS, "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-1000Parcels7Networks_dseg.tsv")
NETS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
LEVELS = ["safe", "low", "medium", "high"]


def play_id(stem):
    m = re.match(r"(sub-\d+)_(ses-\d+)_task-mario_level-(w\dl\d)_rep-(\d+)", stem)
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}_rep-{m.group(4)}" if m else stem


def zrun(x, ok):
    out = np.zeros_like(x, dtype=np.float32)
    if ok.sum() > 1:
        mu, sd = x[ok].mean(0), x[ok].std(0)
        sd[sd == 0] = 1
        out[ok] = (x[ok] - mu) / sd
    return out


def parcel_networks():
    """Yeo-7 network of each Schaefer-1000 parcel (from the atlas label names, header row skipped)."""
    names = []
    for line in open(LABELS):
        names.append(line.rstrip("\n").split("\t")[1])
    net = []
    for n in names[1:]:
        net.append(n.split("_")[2])
    return np.array(net)


def network_means(Y, net):
    """[n, 7]: mean of Y over the parcels of each network in NETS."""
    cols = []
    for n in NETS:
        cols.append(Y[:, net == n].mean(1))
    return np.stack(cols, 1)


def type_counts(typ):
    counts = {}
    for ty in np.unique(typ):
        counts[ty] = int((typ == ty).sum())
    return counts


def load_nuisance(path):
    """-> (W, levels, delays, lowlevel) of a nuisance encoder .npz (results/bold_generator/<sub>/nuis.npz)."""
    nz = np.load(path)
    levels = []
    for l in nz["levels"]:
        levels.append(str(l))
    delays = []
    for d in nz["delays"]:
        delays.append(int(d))
    lowlevel = bool(nz["nuis_lowlevel"]) if "nuis_lowlevel" in nz else False
    return nz["W"], levels, tuple(delays), lowlevel


def nuisance_inputs(z, levels, lowlevel, lowlevel_path):
    """Raw nuisance columns of one run: buttons, level one-hot, time in play (/ 300 s), optionally low-level."""
    onehot = []
    for l in levels:
        onehot.append(z["level"] == l)
    raw = [z["buttons"], np.stack(onehot, 1).astype(np.float32), z["t_play"][:, None] / 300]
    if lowlevel:
        raw.append(np.load(lowlevel_path).astype(np.float32))
    return np.hstack(raw)


def per_level(vals, lv, runs):
    """(mean over runs, SE across runs) of vals at each danger level."""
    out = []
    for L in range(4):
        per_run = []
        for k in np.unique(runs):
            sel = (lv == L) & (runs == k)
            if sel.any():
                per_run.append(vals[sel].mean(0))
        pr = np.array(per_run)
        out.append((pr.mean(0), pr.std(0) / np.sqrt(len(pr))))
    return out


def collect_scans(sub, hold, a):
    """Held-out scans with their peak risk / threat type and the nuisance-free brain response at t+3, t+4."""
    import h5py
    W, levels, delays, lowlevel = load_nuisance(a.nuis or f"{ROOT}/results/bold_generator/{sub}/nuis.npz")
    st = E.scan_table(sub, E.run_trs_from_confounds(sub))
    rows, Ys = [], []
    with h5py.File(E.DERIVED / "parcels" / f"{sub}.h5", "r") as h:
        keys = []
        for k in sorted(h):
            if k.split("_")[0] in hold:
                keys.append(k)
        for key in keys:
            Y = h[key][:].astype(np.float32)
            z = dict(np.load(f"{TRF}/{sub}/{key}.npz"))
            if len(z["coverage"]) != len(Y):
                continue
            inplay = z["coverage"] > 0
            raw = nuisance_inputs(z, levels, lowlevel, f"{TRF}/{sub}/{key}.lowlevel.npy")
            X = E.add_delays(zrun(raw, inplay), delays)
            R = Y - X @ W  # brain signal beyond buttons, level, time, brightness, motion, sound
            g = st[st.key == key].sort_values("tr")
            cache = {}
            for r in g.itertuples():
                t = int(r.tr)
                if r.coverage < 0.999 or not r.stem or t + 4 >= len(Y) or t < E.TRIM:
                    continue
                pid = play_id(r.stem)
                if pid not in cache:
                    p = f"{OUT}/risk/{pid}.npz"
                    cache[pid] = dict(np.load(p)) if os.path.exists(p) else None
                rz = cache[pid]
                if rz is None:
                    continue
                m = (rz["frame"] >= r.f0) & (rz["frame"] < r.f1) & np.isfinite(rz["risk"])
                if not m.any():
                    continue
                horizon = r.f0 + 5 * 1.49 * 60
                if ((rz["death_frame"] >= r.f0) & (rz["death_frame"] < horizon)).any():
                    continue  # a death soon after: leave out (response to dying, not to danger)
                i = np.where(m)[0][np.argmax(rz["risk"][m])]
                rows.append(dict(key=key, tr=t, risk=float(rz["risk"][i]), type=str(rz["type"][i])))
                Ys.append(R[t + 3:t + 5].mean(0))
    return rows, Ys, lowlevel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="sub-03")
    ap.add_argument("--nuis", default=None,
                    help="nuisance encoder .npz (default results/bold_generator/<sub>/nuis.npz)")
    ap.add_argument("--perm", type=int, default=1000)
    a = ap.parse_args()
    sub = a.subject
    hold = set(json.load(open(os.path.join(CODE, "configs", "holdout_sessions.json")))["holdout"][sub])
    net = parcel_networks()
    rows, Ys, lowlevel = collect_scans(sub, hold, a)
    risk, runs, typ = [], [], []
    for row in rows:
        risk.append(row["risk"])
        runs.append(row["key"])
        typ.append(row["type"])
    risk, runs, typ = np.array(risk), np.array(runs), np.array(typ)
    Yr = np.stack(Ys)
    q = np.quantile(risk, [0.5, 0.75, 0.9])
    lvl = np.digitize(risk, q)  # 0 safe .. 3 high
    netY = network_means(Yr, net)  # [scans, 7]

    lev = per_level(netY, lvl, runs)
    diff = netY[lvl == 3].mean(0) - netY[lvl == 0].mean(0)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(a.perm):
        lp = lvl.copy()
        for k in np.unique(runs):
            idx = np.where(runs == k)[0]
            lp[idx] = rng.permutation(lp[idx])
        null.append(netY[lp == 3].mean(0) - netY[lp == 0].mean(0))
    null = np.array(null)
    p = ((np.abs(null) >= np.abs(diff)).sum(0) + 1) / (a.perm + 1)
    hi, lo = Yr[lvl == 3], Yr[lvl == 0]
    t_parcel = (hi.mean(0) - lo.mean(0)) / np.sqrt(hi.var(0) / len(hi) + lo.var(0) / len(lo))
    sel = lvl >= 2
    by_type = {}
    for ty in ("enemy", "hole"):
        if (sel & (typ == ty)).sum() > 5:
            by_type[ty] = dict(n=int((sel & (typ == ty)).sum()), mean=netY[sel & (typ == ty)].mean(0).tolist())
    n_per_level, mean_risk, response, se = [], [], [], []
    for L in range(4):
        n_per_level.append(int((lvl == L).sum()))
        mean_risk.append(float(risk[lvl == L].mean()))
        response.append(lev[L][0].tolist())
        se.append(lev[L][1].tolist())
    res = dict(subject=sub, held_out_sessions=sorted(hold), n_scans=int(len(rows)),
               n_per_level=n_per_level, risk_thresholds=q.tolist(),
               mean_risk_per_level=mean_risk, networks=NETS, levels=LEVELS,
               response=response, se=se, high_minus_safe=diff.tolist(),
               p_perm=p.tolist(), by_type_medium_high=by_type,
               type_counts=type_counts(typ), nuisance_lowlevel=lowlevel)
    json.dump(res, open(f"{OUT}/human_{sub}.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/human_{sub}_parcels.npz", t_high_minus_safe=t_parcel.astype(np.float32),
                        diff=(hi.mean(0) - lo.mean(0)).astype(np.float32))
    print(f"{sub}: {len(rows)} scans (held-out), per level {res['n_per_level']}, risk thresholds "
          f"{np.round(q, 3).tolist()}, mean risk per level {np.round(res['mean_risk_per_level'], 3).tolist()}")
    header = []
    for L in LEVELS:
        header.append(f"{L:>14s}")
    print("network      " + "  ".join(header) + "   high-safe   p")
    for j, n in enumerate(NETS):
        cells = []
        for L in range(4):
            cells.append(f"{lev[L][0][j]:+.3f}±{lev[L][1][j]:.3f}")
        print(f"{n:12s} " + "  ".join(cells) + f"   {diff[j]:+.3f}   {p[j]:.3f}")
    by_type_short = {}
    for k, v in by_type.items():
        by_type_short[k] = (v["n"], np.round(v["mean"], 3).tolist())
    print("threat type (medium + high):", by_type_short)
    print("parcels with t > 3:", int((t_parcel > 3).sum()), "| t < -3:", int((t_parcel < -3).sum()))


if __name__ == "__main__":
    main()
