"""Encoding models for CNeuroMod Mario, tested on the held-out fMRI sessions (CPU).

BOLD: Schaefer-1000 parcel time series (cneuromod/derived/parcels, read-only). Features: build_tr_features.py.
Bands (per scan; z-scored per run; wide bands reduced by PCA fitted on training runs only):
  nuis        buttons (12) + level one-hot + time in play     in every model (motor, level, drift), as
                                                              Reason-to-Play's nuisance bands
              (--lowlevel: + luminance, optical flow, audio envelope, CNeuroMod's low-level confounds;
              results then go to <sub>_lowlevel/ and summary_<subjects>_lowlevel.csv)
  gamestate   13 RAM variables (Mario motion / pose, enemies)
  vjepa47     V-JEPA 2.1 G block 47                (PCA 256)
  vjepa4      the 4 blocks 11/23/37/47 together     (PCA 256)
  agent_<tag>_d<j>  the agent's internal state at depth j (layers 13/26/39/52) from brain_pred/agent/embed_scans.py,
              or the running context of its replies from brain_pred/agent/context_features.py (PCA 256)
Models: nuis; nuis + each band; nuis + vjepa47 + gamestate; and the control nuis + vjepa47_shift (the band
circularly shifted by half a run inside each run: same statistics, timing destroyed).
Two-band ridge (mjg.encoding.fit_episodes_banded): nuisance and the feature band get separate penalties, chosen
per parcel by CV, and the band can be dropped (scale 0). FIR delays 1-5 TRs within run (1.5-7.5 s). Scans used:
fully inside one play, not the first / last 5 of a run, with V-JEPA rows (same scans for every model). Train: runs
outside configs/holdout_sessions.json, alpha per parcel by play-grouped 5-fold CV (closed form). Test: the held-out
sessions' runs. Metric: Pearson r per parcel; gain = r(model) - r(nuis). No noise ceiling (no repeated stimuli in
Mario).
Model names: bands joined by '+' share one penalty; groups joined by '&' get one penalty each
(mjg.encoding.fit_episodes_multiband), e.g. 'vjepa4&agent_sftfresh_d1'. Any such name works in --models.
Final model: 'vjepa4&agent_<state tag>_d1&agent_<context tag>_d1' (V-JEPA, the agent's state, its reply context; one
penalty each). Control: agent_<tag>_d<j>_shift = the band circularly shifted by half a run (timing destroyed).

  python encode.py --subjects sub-01   # -> $MJG_ROOT/results/brain_encoding/<sub>/<model>.npz, summary_<subjects>.csv
  --save DIR: also write each fitted model in a reusable form, DIR/<sub>/<model>.npz (weights W [p, 1000], FIR delays,
  band list, level list, PCA mean / components per wide band, held-out r per parcel). brain_pred/generator
  applies it to new gameplay (a person's or the agent's) to generate BOLD.
"""
import argparse, json, os, re, sys, time

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, REPO)
from mjg import encoding as E  # noqa: E402

ROOT = os.environ.get("MJG_ROOT", ".")
TRF = f"{ROOT}/results/brain_encoding/tr_features"
OUT = f"{ROOT}/results/brain_encoding"
DELAYS = (1, 2, 3, 4, 5)
N_PC = 256
MODELS = {
    "nuis": [],
    "gamestate": ["gamestate"],
    "vjepa47": ["vjepa47"],
    "vjepa4": ["vjepa4"],
    "control_vjepa47_shift": ["vjepa47_shift"],
}
WIDE = {"vjepa47", "vjepa4", "vjepa47_shift"}
AGENT_BAND = re.compile(r"^agent_(.+)_d(\d)(?:_(shift))?$")


def zrun(x, ok):
    """z-score a run's band on the scans inside a play; scans outside a play set to 0."""
    out = np.zeros_like(x, dtype=np.float32)
    if ok.sum() > 1:
        mu, sd = x[ok].mean(0), x[ok].std(0)
        sd[sd == 0] = 1
        out[ok] = (x[ok] - mu) / sd
    return out


def select_models(a):
    """--models / --agent-tags / --combine -> {model name: bands}; also marks the agent bands as wide (PCA)."""
    for t in a.agent_tags:
        for j in range(4):
            MODELS[f"agent_{t}_d{j}"] = [f"agent_{t}_d{j}"]
            WIDE.add(f"agent_{t}_d{j}")
    for c in a.combine:
        MODELS[c] = c.split("+")
    if a.models == "all":
        models = MODELS
    else:
        models = {}
        for m in a.models.split(","):
            models[m] = MODELS.get(m) or m.replace("&", "+").split("+")
    for bs in models.values():
        for b in bs:
            g = AGENT_BAND.match(b)
            if g and g.group(1) in a.agent_tags:
                WIDE.add(b)
    return models


def penalty_groups(models):
    """model name -> its penalty groups: one per '&' part (bands joined by '+' inside a part), else all its bands."""
    groups = {}
    for m, bs in models.items():
        if "&" in m:
            parts = []
            for part in m.split("&"):
                parts.append(part.split("+"))
            groups[m] = parts
        else:
            groups[m] = [bs]
    return groups


def load_run_features(sub, k, n_bold, a):
    """Per-scan features of one run (+ agent states, + low-level confounds), or None when something is missing."""
    p = f"{TRF}/{sub}/{k}.npz"
    if not os.path.exists(p):
        return None
    z = dict(np.load(p))
    for t in a.agent_tags:  # the agent's internal state per scan [T, 4, hidden] + mask
        ap_ = p.replace(".npz", f".agent_{t}.npy")
        if not os.path.exists(ap_):
            print(f"  skip {k}: no agent states yet")
            return None
        z[f"agent_{t}"] = np.load(ap_)
        z[f"agent_{t}_mask"] = np.load(ap_.replace(".npy", ".mask.npy"))
    if a.lowlevel:
        lp = p.replace(".npz", ".lowlevel.npy")
        if not os.path.exists(lp):
            print(f"  skip {k}: no low-level confounds yet")
            return None
        z["lowlevel"] = np.load(lp).astype(np.float32)
    if len(z["coverage"]) != n_bold:
        print(f"  skip {k}: {len(z['coverage'])} feature scans vs {n_bold} BOLD scans")
        return None
    return z


def load_runs(sub, a):
    """-> BOLD {run: [T, 1000]}, features {run: dict}, the runs that have both (sorted)."""
    import h5py
    Y = {}
    with h5py.File(E.DERIVED / "parcels" / f"{sub}.h5", "r") as h:
        for k in h:
            Y[k] = h[k][:].astype(np.float32)
    F, keys = {}, []
    for k in sorted(Y):
        z = load_run_features(sub, k, len(Y[k]), a)
        if z is not None:
            F[k] = z
            keys.append(k)
    return Y, F, keys


def raw_band(z, b, levels, lowlevel):
    """One band of one run, before z-scoring: [T, p]."""
    if b == "nuis":
        one_hot = []
        for l in levels:
            one_hot.append(z["level"] == l)
        lv = np.stack(one_hot, 1).astype(np.float32)
        parts = [z["buttons"], lv, z["t_play"][:, None] / 300]
        if lowlevel:
            parts.append(z["lowlevel"])
        return np.hstack(parts)
    if b == "gamestate":
        return z["gamestate"]
    if b in ("vjepa47", "vjepa47_shift"):
        x = z["vjepa"][:, 3].astype(np.float32)  # block 47
        return np.roll(x, len(x) // 2, 0) if b == "vjepa47_shift" else x
    if b.startswith("agent_"):
        tag, d, var = AGENT_BAND.match(b).groups()
        x = z[f"agent_{tag}"][:, int(d)].astype(np.float32)
        return np.roll(x, len(x) // 2, 0) if var == "shift" else x
    if b == "vjepa4":
        return z["vjepa"].reshape(len(z["vjepa"]), -1).astype(np.float32)
    return z[b]


def scans_used(F, keys, agent_tags):
    """-> scans used for fitting / testing {run: bool [T]}, scans inside a play {run: bool [T]}."""
    keep, inplay = {}, {}
    for k in keys:
        z = F[k]
        T = len(z["coverage"])
        inplay[k] = z["coverage"] > 0
        tr = np.arange(T)
        keep[k] = (z["coverage"] >= 0.999) & (tr >= E.TRIM) & (tr < T - E.TRIM) & (z["n_rows"] > 0)
        for t in agent_tags:  # same scans for every model: those with an agent state
            keep[k] &= z[f"agent_{t}_mask"] if f"agent_{t}_mask" in z else np.zeros(T, bool)
    return keep, inplay


def build_design(sub, bands_needed, F, keys, train_keys, keep, inplay, levels, lowlevel):
    """Every needed band of every run, z-scored per run, wide bands through a PCA fitted on the training scans."""
    design, reducers = {}, {}
    for b in sorted(bands_needed):
        per = {}
        for k in keys:
            per[k] = zrun(raw_band(F[k], b, levels, lowlevel), inplay[k])
        if b in WIDE:
            train_rows = []
            for k in train_keys:
                train_rows.append(per[k][keep[k]])
            red = E.PCAReducer(N_PC).fit(train_rows)
            reducers[b] = red
            for k in keys:
                per[k] = np.where(inplay[k][:, None], red.transform(per[k]), 0).astype(np.float32)
            print(f"  {sub} {b}: PCA {N_PC} explains {red.explained_:.2f}", flush=True)
        design[b] = per
    return design, reducers


def fit_one(bands, groups, design, F, Y, keys, train_keys, test_keys, keep):
    """nuis + bands: FIR design, ridge fitted on the training plays, Pearson r per parcel on the held-out runs."""
    X = {}
    for k in keys:
        cols = []
        for b in ["nuis"] + bands:
            cols.append(design[b][k])
        X[k] = E.add_delays(np.hstack(cols), DELAYS)
    k0 = keys[0]
    n_nuis, p_one = design["nuis"][k0].shape[1], X[k0].shape[1] // len(DELAYS)
    band_mask = (np.arange(X[k0].shape[1]) % p_one) >= n_nuis  # feature-band columns of every delay
    ids = [np.zeros(n_nuis, int)]
    for gi, g in enumerate(groups):
        for b in g:
            ids.append(np.full(design[b][k0].shape[1], gi + 1))
    ids_one = np.concatenate(ids)
    Xep, Yep = [], []
    for k in train_keys:
        st = F[k]["stem"]
        for s in np.unique(st[keep[k]]):
            m = keep[k] & (st == s)
            Xep.append(X[k][m])
            Yep.append(Y[k][m])
    if len(groups) > 1:  # one penalty per '&' group
        W, alphas, scales = E.fit_episodes_multiband(Xep, Yep, np.tile(ids_one, len(DELAYS)))
    elif bands:
        W, alphas, scales = E.fit_episodes_banded(Xep, Yep, band_mask)
    else:
        (W, alphas), scales = E.fit_episodes(Xep, Yep), np.zeros(Y[k0].shape[1])
    Xs, Ys = [], []
    for k in test_keys:
        Xs.append(X[k][keep[k]])
        Ys.append(Y[k][keep[k]])
    Xte, Yte = np.concatenate(Xs), np.concatenate(Ys)
    r = E.pearson_cols(Xte @ W.astype(np.float32), Yte)
    return dict(W=W, alphas=alphas, scales=scales, r=r, n_nuis=n_nuis, n_features=Xte.shape[1], n_plays=len(Xep))


def group_usage(scales, n_parcels):
    """-> parcels where any feature group is used, and '/'-joined parcel counts per group."""
    used = np.asarray(scales).reshape(n_parcels, -1) > 0
    counts = []
    for c in used.sum(0):
        counts.append(str(int(c)))
    return int(used.any(1).sum()), "/".join(counts)


def save_encoder(path, fit, bands, reducers, levels, gain, lowlevel, held_out):
    """The fitted model in the reusable form brain_pred/prediction/predict.py reads."""
    extra = {}
    for b in bands:
        if b in reducers:
            extra[f"pca_mean__{b}"] = reducers[b].mean_.astype(np.float32)
            extra[f"pca_comps__{b}"] = reducers[b].comps_.astype(np.float32)
    np.savez_compressed(path, W=fit["W"].astype(np.float32), alphas=fit["alphas"].astype(np.float32),
                        band_scale=np.asarray(fit["scales"], np.float32), bands=np.array(["nuis"] + bands),
                        levels=np.array(levels), delays=np.array(DELAYS), n_nuis=fit["n_nuis"],
                        nuis_lowlevel=bool(lowlevel),
                        r_heldout=fit["r"].astype(np.float32), gain_heldout=gain.astype(np.float32),
                        held_out_sessions=np.array(held_out), tr=np.float32(1.49), **extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", default=["sub-01"])
    ap.add_argument("--models", default="all")
    ap.add_argument("--save", default=None, help="write reusable fitted models to this folder (see docstring)")
    ap.add_argument("--agent-tags", nargs="*", default=[],
                    help="agent internal-state bands (brain_pred/agent/embed_scans.py --tag): adds models "
                         "agent_<tag>_d0..d3 (4 depths) and restricts every model to scans with an agent state, so "
                         "all compare on the same scans")
    ap.add_argument("--summary-tag", default="", help="appended to the summary csv name (parallel runs on one subject)")
    ap.add_argument("--combine", nargs="*", default=[],
                    help="joint models, bands joined by '+' (e.g. vjepa4+agent_sftfresh_d1): one feature band, "
                         "one penalty")
    ap.add_argument("--folder-tag", default="",
                    help="appended to the per-model results folder (keeps reference npz files)")
    ap.add_argument("--lowlevel", action="store_true",
                    help="add the low-level confounds (build_tr_features.py --extra lowlevel) to the nuisance band")
    a = ap.parse_args()
    import pandas as pd
    hold = json.load(open(os.path.join(REPO, "configs", "holdout_sessions.json")))["holdout"]
    models = select_models(a)
    groups = penalty_groups(models)
    rows, sfx = [], ("_lowlevel" if a.lowlevel else "") + a.folder_tag
    summary = f"{OUT}/summary_{'_'.join(a.subjects)}{sfx}{('_' + a.summary_tag) if a.summary_tag else ''}.csv"
    for sub in a.subjects:
        t0 = time.time()
        Y, F, keys = load_runs(sub, a)
        level_set = set()
        for k in keys:
            for l in F[k]["level"]:
                if l:
                    level_set.add(l)
        levels = sorted(level_set)
        held = set(hold.get(sub, []))
        test_keys, train_keys = [], []
        for k in keys:
            if k.split("_")[0] in held:
                test_keys.append(k)
            else:
                train_keys.append(k)
        keep, inplay = scans_used(F, keys, a.agent_tags)
        bands_needed = {"nuis"}
        for bs in models.values():
            for b in bs:
                bands_needed.add(b)
        design, reducers = build_design(sub, bands_needed, F, keys, train_keys, keep, inplay, levels, a.lowlevel)
        n_tr = 0
        for k in train_keys:
            n_tr += int(keep[k].sum())
        n_te = 0
        for k in test_keys:
            n_te += int(keep[k].sum())
        print(f"{sub}: {len(keys)} runs ({len(train_keys)} train / {len(test_keys)} test, held-out sessions "
              f"{hold.get(sub)}); scans used {n_tr} train / {n_te} test; levels {len(levels)} "
              f"({time.time() - t0:.0f}s)", flush=True)
        os.makedirs(f"{OUT}/{sub}{sfx}", exist_ok=True)
        r_nuis = None
        for name, bands in models.items():
            have_all = True
            for b in bands:
                if b not in design:
                    have_all = False
                    break
            if not have_all:
                print(f"  skip {name}: band missing")
                continue
            t1 = time.time()
            fit = fit_one(bands, groups[name], design, F, Y, keys, train_keys, test_keys, keep)
            r, alphas, scales = fit["r"], fit["alphas"], fit["scales"]
            if name == "nuis":
                r_nuis = r
            gain = r - r_nuis if r_nuis is not None else np.zeros_like(r)
            np.savez_compressed(f"{OUT}/{sub}{sfx}/{name}.npz", r=r.astype(np.float32), gain=gain.astype(np.float32),
                                alphas=alphas.astype(np.float32), band_scale=np.asarray(scales, np.float32))
            if a.save:
                os.makedirs(f"{a.save}/{sub}", exist_ok=True)
                save_encoder(f"{a.save}/{sub}/{name}.npz", fit, bands, reducers, levels, gain, a.lowlevel,
                             hold.get(sub, []))
            n_used, each_used = group_usage(scales, len(alphas))
            rows.append(dict(subject=sub, model=name, n_features=fit["n_features"], n_train_plays=fit["n_plays"],
                             n_test_scans=n_te, mean_r=float(np.nanmean(r)), top100_r=float(np.sort(r)[-100:].mean()),
                             mean_gain=float(np.nanmean(gain)), top100_gain=float(np.sort(gain)[-100:].mean()),
                             parcels_gain_pos=int((gain > 0).sum()), median_alpha=float(np.median(alphas)),
                             parcels_band_used=n_used, parcels_each_group_used=each_used,
                             sec=round(time.time() - t1)))
            print(f"  {sub} {name:24s} mean r {rows[-1]['mean_r']:.4f}  top100 {rows[-1]['top100_r']:.3f}  "
                  f"gain {rows[-1]['mean_gain']:+.4f} (top100 {rows[-1]['top100_gain']:+.3f})  ({rows[-1]['sec']}s)",
                  flush=True)
            pd.DataFrame(rows).to_csv(summary, index=False)  # one file per job


if __name__ == "__main__":
    main()
