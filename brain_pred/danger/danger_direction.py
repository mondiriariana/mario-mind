"""Danger direction (CPU): does a model's own sense of danger line up with what predicts the human brain's danger
regions?

No training of the agent, no GPU. For one subject (default sub-03), per model and layer, on the same scans:
  (a) danger readout: ridge from the model's state at scan t to the danger of scan t (peak risk in the scan,
      risk_model.py: chance Mario dies within 3 s), as human_response.py defines it.
  (b) brain readout: ridge from the same state to the human brain's danger regions 4.5-7.5 s later (mean of scans t+3,
      t+4), as human_response.py: parcel BOLD minus the nuisance encoder's prediction (results/bold_generator/<sub>/
      nuis.npz). Target = mean over the parcel set, and each parcel separately.
  (c) do they line up: held-out correlation between the outputs of (a) and (b); cosine of the two weight vectors in the
      PCA space (each PC z-scored per run, so the space is whitened: the raw-weight cosine would depend on the feature
      covariance); does (a) alone predict the parcel-set BOLD. Nulls: (a) refit on the danger series circularly shifted
      inside each training run (timing destroyed, same statistics), --perm times, with its alpha kept.
  By threat type (held-out, medium + high danger as human_response.py): mean real BOLD, (a) and (b) for gaps vs enemies,
  relative to safe scans.
Scans: fully inside a play, not in the first 5 of a run, no death within 5 scans (human_response.py rules), with a state
for every agent tag (same scans for every model). Fit on the training sessions (alpha per target by play-grouped CV,
mjg.encoding.fit_episodes; features and targets z-scored per run), scored on the held-out sessions
(configs/holdout_sessions.json).
Parcel sets: "human88" = results/danger/human_<sub>_parcels.npz t > 3 (chosen on the held-out sessions, so the brain
readout's held-out score there is optimistic), and "train_sel" = the same high - safe contrast computed on the training
sessions only, t > 3 (clean).
Models: nuisance (buttons, level, time in play, low-level: a floor), V-JEPA (vjepa4, PCA 256), agent_<tag>_d0..d3
(layers 13 / 26 / 39 / 52, PCA 256). For the models in --save-dirs, the danger readout is also saved as a direction in
the raw hidden space (results/danger/direction/<sub>_<model>_danger_dir.npz, for steer_play.py).
Writes results/danger/direction/<sub>.json and prints a table.

  $MJG_SCI_PY danger_direction.py --subject sub-03 --tags sft plain      (CPU node)
"""
import argparse, json, os, sys, time

import numpy as np

ROOT = os.environ.get("MJG_ROOT", ".")
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "..")
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)
from mjg import encoding as E  # noqa: E402
from human_response import TRF, OUT, load_nuisance, nuisance_inputs, play_id, zrun  # noqa: E402

N_PC = 256


def zsel(x, ok):
    """z-score the rows in ok (per run); other rows 0."""
    out = np.zeros_like(x, dtype=np.float32)
    if ok.sum() > 1:
        mu, sd = x[ok].mean(0), x[ok].std(0)
        sd = np.where(sd == 0, 1, sd)
        out[ok] = (x[ok] - mu) / sd
    return out


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def stack(runs, D, ks):
    """Concatenate D[k] over the runs ks, on each run's used scans."""
    parts = []
    for k in ks:
        parts.append(D[k][runs[k]["ok"]])
    return np.concatenate(parts)


def run_field(runs, field, ps=None):
    """{run key: runs[key][field]} (or runs[key][field][ps] for the per-parcel-set targets)."""
    out = {}
    for k in runs:
        out[k] = runs[k][field] if ps is None else runs[k][field][ps]
    return out


def episodes(runs, trk, X, Yk):
    """Training data split into plays (the CV groups of fit_episodes)."""
    Xe, Ye = [], []
    for k in trk:
        ok, s = runs[k]["ok"], runs[k]["stem"]
        for pl in np.unique(s[ok]):
            m = ok & (s == pl)
            Xe.append(X[k][m])
            Ye.append(Yk[k][m])
    return Xe, Ye


def contrast(runs, ks):
    """Per-parcel t of high - safe danger on the runs ks (levels from the risk quantiles of those scans)."""
    rk = stack(runs, run_field(runs, "risk"), ks)
    Bk = stack(runs, run_field(runs, "bold"), ks)
    q = np.quantile(rk, [0.5, 0.75, 0.9])
    lvl = np.digitize(rk, q)
    hi, lo = Bk[lvl == 3], Bk[lvl == 0]
    return (hi.mean(0) - lo.mean(0)) / np.sqrt(hi.var(0) / len(hi) + lo.var(0) / len(lo))


def save_direction(a, sub, name, runs, trk, X, Wa, red):
    """The danger readout as a direction in the layer's raw hidden space, three ways (unit vectors):
      pattern  covariance of the raw state with the readout's output on the training scans (Haufe et al. 2014: the
               direction the states actually move along when the readout rises; the one used for steering)
      filter   the readout's gradient w.r.t. the raw state (PCA weights mapped back, divided by the feature spread)
      diffmean mean raw state of the top-10% danger scans minus the safe half (contrastive, model-free)
    Raw states are centred per run (as the readout sees them). Also: sd of the training states' projection on each
    direction (the steering unit) and the mean norm of the raw state."""
    raw, norms, x_tr, risk, zr = [], [], [], [], []
    for k in trk:
        ok, f = runs[k]["ok"], runs[k]["feats"][name]
        raw.append((f - f[ok].mean(0))[ok])
        norms.append(np.linalg.norm(f[ok], axis=1).mean())
        x_tr.append(X[k][ok])
        risk.append(runs[k]["risk"][ok])
        zr.append(E._z(f[ok]))
    raw = np.concatenate(raw).astype(np.float32)
    norm = float(np.mean(norms))
    a_tr = np.concatenate(x_tr) @ Wa[:, 0]
    risk = np.concatenate(risk)
    zr = np.concatenate(zr)
    pattern = raw.T @ (a_tr - a_tr.mean()) / len(a_tr)
    sd_raw = raw.std(0)
    sd_raw[sd_raw == 0] = 1
    sd_pc = ((zr - red.mean_) @ red.comps_).std(0)
    sd_pc[sd_pc == 0] = 1
    filt = (red.comps_ @ (Wa[:, 0] / sd_pc)) / sd_raw
    hi, lo = risk >= np.quantile(risk, 0.9), risk < np.quantile(risk, 0.5)
    diffm = raw[hi].mean(0) - raw[lo].mean(0)
    dirs = {"pattern": unit(pattern), "filter": unit(filt), "diffmean": unit(diffm)}
    sd, sd_keys = {}, {}
    for k, v in dirs.items():
        sd[k] = float((raw @ v).std())
        sd_keys[f"sd_{k}"] = sd[k]
    cos = {}
    for x, y in (("pattern", "filter"), ("pattern", "diffmean"), ("filter", "diffmean")):
        cos[f"{x}_{y}"] = float(dirs[x] @ dirs[y])
    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/{sub}_{name}_danger_dir.npz", **dirs, **sd_keys, mean_state_norm=norm, depth=int(name[-1]),
             n_train_scans=len(raw))
    sd_txt, cos_txt = [], []
    for k, v in sd.items():
        sd_txt.append(f"{k} {v:.2f}")
    for k, v in cos.items():
        cos_txt.append(f"{k} {v:+.2f}")
    print(f"  saved {name} danger direction: projection sd {', '.join(sd_txt)}; "
          f"mean state norm {norm:.1f}; cosines {', '.join(cos_txt)}", flush=True)


def scan_danger(g, T, masks, cache):
    """Per scan of one run: used or not (fully in a play, a state for every tag, no death soon after), peak risk and
    threat type."""
    ok = np.zeros(T, bool)
    risk = np.zeros(T, np.float32)
    typ = np.array([""] * T, dtype=object)
    for r in g.itertuples():
        t = int(r.tr)
        if r.coverage < 0.999 or not r.stem or t + 4 >= T or t < E.TRIM:
            continue
        has_state = True
        for m in masks.values():
            if not m[t]:
                has_state = False
                break
        if not has_state:
            continue
        pid = play_id(r.stem)
        if pid not in cache:
            pp = f"{OUT}/risk/{pid}.npz"
            cache[pid] = dict(np.load(pp)) if os.path.exists(pp) else None
        rz = cache[pid]
        if rz is None:
            continue
        m = (rz["frame"] >= r.f0) & (rz["frame"] < r.f1) & np.isfinite(rz["risk"])
        if not m.any():
            continue
        if ((rz["death_frame"] >= r.f0) & (rz["death_frame"] < r.f0 + 5 * 1.49 * 60)).any():
            continue
        i = np.where(m)[0][np.argmax(rz["risk"][m])]
        ok[t], risk[t], typ[t] = True, rz["risk"][i], str(rz["type"][i])
    return ok, risk, typ


def load_runs(a, sub, hold):
    """Per run: used scans, danger, threat type, nuisance-free BOLD at t+3/t+4, raw features of every model."""
    import h5py
    Wn, levels, delays, lowlevel = load_nuisance(f"{ROOT}/results/bold_generator/{sub}/nuis.npz")
    st = E.scan_table(sub, E.run_trs_from_confounds(sub))
    runs = {}
    with h5py.File(E.DERIVED / "parcels" / f"{sub}.h5", "r") as h:
        keys = sorted(h)
        tr_k, te_k = [], []
        for k in keys:
            if k.split("_")[0] in hold:
                te_k.append(k)
            else:
                tr_k.append(k)
        if a.max_runs:
            keys = tr_k[:a.max_runs] + te_k[:max(1, a.max_runs // 2)]
        for key in keys:
            p = f"{TRF}/{sub}/{key}.npz"
            ag = {}
            for t in a.tags:
                ag[t] = f"{TRF}/{sub}/{key}.agent_{t}.npy"
            have_all = os.path.exists(p)
            for x in ag.values():
                have_all = have_all and os.path.exists(x)
            if not have_all:
                continue
            Y = h[key][:].astype(np.float32)
            z = dict(np.load(p))
            if len(z["coverage"]) != len(Y):
                continue
            inplay = z["coverage"] > 0
            nuis_raw = nuisance_inputs(z, levels, lowlevel, f"{TRF}/{sub}/{key}.lowlevel.npy")
            # brain signal beyond hands, level, time, brightness, motion, sound
            R = Y - E.add_delays(zrun(nuis_raw, inplay), delays) @ Wn
            masks = {}
            for t in a.tags:
                masks[t] = np.load(ag[t].replace(".npy", ".mask.npy"))
            g = st[st.key == key].sort_values("tr")
            T = len(Y)
            ok, risk, typ = scan_danger(g, T, masks, {})
            if ok.sum() < 10:
                continue
            bold = np.zeros_like(R)
            bold[:T - 4] = 0.5 * (R[3:T - 1] + R[4:T])  # scan t -> mean of t+3, t+4
            feats = {"nuisance": nuis_raw, "vjepa": z["vjepa"].reshape(T, -1).astype(np.float32)}
            for t in a.tags:
                x = np.load(ag[t])
                for d in range(x.shape[1]):
                    feats[f"agent_{t}_d{d}"] = x[:, d].astype(np.float32)
            stem = np.array([""] * T, dtype=object)
            stem[g.tr.values.astype(int)[g.tr.values < T]] = g.stem.values[g.tr.values < T]
            runs[key] = dict(ok=ok, inplay=inplay, risk=risk, typ=typ, bold=bold, feats=feats, stem=stem,
                             test=key.split("_")[0] in hold)
    return runs, lowlevel


def model_features(runs, trk, ok_tr, name):
    """Per run: the model's features z-scored per run, then PCA (fit on the training scans) except for the nuisance."""
    X = {}
    for k in runs:
        X[k] = zrun(runs[k]["feats"][name], runs[k]["inplay"])
    if name == "nuisance":
        for k in runs:
            X[k] = zsel(X[k], runs[k]["ok"])
        return X, None, None
    fit_rows = []
    for k, o in zip(trk, ok_tr):
        fit_rows.append(X[k][o])
    red = E.PCAReducer(N_PC).fit(fit_rows)
    for k in runs:
        X[k] = red.transform(X[k])
    return X, red, red.explained_


def shifted_nulls(runs, trk, X, Xte, Wa, alpha_a, rng, n_perm):
    """Held-out outputs of the danger readout refit on danger series circularly shifted inside each training run."""
    Xe, _ = episodes(runs, trk, X, run_field(runs, "risk_z"))
    Xtr_all = np.concatenate(Xe)
    G = Xtr_all.T.astype(np.float64) @ Xtr_all
    lam, V = np.linalg.eigh(G)
    lam = np.clip(lam, 0, None)
    null_a = []
    for _ in range(n_perm):
        sh = {}
        for k in trk:
            o = runs[k]["ok"]
            y = runs[k]["risk_z"].copy()
            y[o] = np.roll(y[o], rng.integers(len(y[o]) // 4, 3 * len(y[o]) // 4))
            sh[k] = y
        _, Ye = episodes(runs, trk, X, sh)
        C = Xtr_all.T.astype(np.float64) @ np.concatenate(Ye)
        null_a.append((Xte @ (V @ ((V.T @ C) / (lam[:, None] + alpha_a[None, :]))).astype(np.float32))[:, 0])
    return null_a


def p_two_sided(null, obs, n_perm):
    return float(((np.abs(null) >= abs(obs)).sum() + 1) / (n_perm + 1))


def brain_readout(runs, trk, tek, X, Xte, Wa, a_hat, null_a, groups, ps, n_perm):
    """Fit (b) for parcel set ps and compare it with the danger readout (a)."""
    Yk = run_field(runs, "bold_z", ps)
    Wb, alpha_b = E.fit_episodes(*episodes(runs, trk, X, Yk))
    Yte = stack(runs, Yk, tek)
    b_all = Xte @ Wb
    b_hat = b_all[:, 0]
    r_par = E.pearson_cols(b_all[:, 1:], Yte[:, 1:])
    wa, wb = Wa[:, 0], Wb[:, 0]
    cos = float(wa @ wb / np.sqrt((wa @ wa) * (wb @ wb)))
    ab = corr(a_hat, b_hat)
    a_bold = corr(a_hat, Yte[:, 0])
    n_ab, n_ad = [], []
    for n in null_a:
        n_ab.append(corr(n, b_hat))
        n_ad.append(corr(n, Yte[:, 0]))
    n_ab, n_ad = np.array(n_ab), np.array(n_ad)
    by_group = {}
    for gname, gmask in groups.items():
        by_group[gname] = dict(danger_readout=float(a_hat[gmask].mean()), brain_readout=float(b_hat[gmask].mean()))
    return dict(
        r_brain=corr(b_hat, Yte[:, 0]), r_brain_per_parcel_mean=float(np.mean(r_par)), alpha_brain=float(alpha_b[0]),
        r_between_readouts=ab, p_between=p_two_sided(n_ab, ab, n_perm), null_between_sd=float(n_ab.std()),
        r_danger_readout_to_bold=a_bold, p_danger_to_bold=p_two_sided(n_ad, a_bold, n_perm),
        cos_whitened=cos, by_group=by_group)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="sub-03")
    ap.add_argument("--tags", nargs="+", default=["sft", "plain"])
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--max-runs", type=int, default=0,
                    help="only the first N training runs + half as many held-out runs (0 = all)")
    ap.add_argument("--out", default=f"{OUT}/direction")
    ap.add_argument("--save-dirs", nargs="*", default=["agent_sft_d1", "agent_sft_d0"],
                    help="models whose danger direction is saved in the raw hidden space (for steering)")
    a = ap.parse_args()
    t0 = time.time()
    sub = a.subject
    hold = set(json.load(open(os.path.join(CODE, "configs", "holdout_sessions.json")))["holdout"][sub])
    t88 = np.load(f"{OUT}/human_{sub}_parcels.npz")["t_high_minus_safe"]

    # ---------------------------------------------------------- scans, danger, brain signal, raw features per run
    runs, lowlevel = load_runs(a, sub, hold)
    trk, tek = [], []
    n_tr, n_te = 0, 0
    for k in runs:
        if runs[k]["test"]:
            tek.append(k)
            n_te += int(runs[k]["ok"].sum())
        else:
            trk.append(k)
            n_tr += int(runs[k]["ok"].sum())
    print(f"{sub}: {len(trk)} training runs ({n_tr} scans) / {len(tek)} held-out runs ({n_te} scans); "
          f"low-level nuisance {lowlevel} ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ parcel sets
    t_train = contrast(runs, trk)
    psets = {"human88": t88 > 3, "train_sel": t_train > 3}
    overlap = int((psets["human88"] & psets["train_sel"]).sum())
    print(f"parcel sets: human88 {int(psets['human88'].sum())}, train_sel {int(psets['train_sel'].sum())} "
          f"(overlap {overlap}); corr of the two t maps {corr(t88, t_train):.3f}", flush=True)

    # targets z-scored per run on the used scans
    for k in runs:
        R_ = runs[k]
        R_["risk_z"] = zsel(R_["risk"][:, None], R_["ok"])
        R_["bold_z"] = {}
        for ps, m in psets.items():  # column 0 = set mean, then each parcel
            R_["bold_z"][ps] = zsel(np.hstack([R_["bold"][:, m].mean(1, keepdims=True), R_["bold"][:, m]]), R_["ok"])
    ok_tr = []
    for k in trk:
        ok_tr.append(runs[k]["ok"])

    # held-out real danger vs brain (model-free reference)
    risk_te = stack(runs, run_field(runs, "risk_z"), tek)[:, 0]
    ref, ref_txt = {}, []
    for ps in psets:
        ref[ps] = corr(risk_te, stack(runs, run_field(runs, "bold_z", ps), tek)[:, 0])
        ref_txt.append(f"{ps} {ref[ps]:+.3f}")
    print("reference, held-out r(real danger, region BOLD): " + ", ".join(ref_txt), flush=True)
    risk_raw_te = stack(runs, run_field(runs, "risk"), tek)
    typ_te = stack(runs, run_field(runs, "typ"), tek)
    q = np.quantile(risk_raw_te, [0.5, 0.75, 0.9])
    lvl_te = np.digitize(risk_raw_te, q)
    groups = {"safe": lvl_te == 0, "gap (medium+high)": (lvl_te >= 2) & (typ_te == "hole"),
              "enemy (medium+high)": (lvl_te >= 2) & (typ_te == "enemy")}

    rng = np.random.default_rng(0)
    models = ["nuisance", "vjepa"]
    for t in a.tags:
        for d in range(4):
            models.append(f"agent_{t}_d{d}")
    parcel_sets, group_n, real_bold = {}, {}, {}
    for ps, m in psets.items():
        parcel_sets[ps] = int(m.sum())
    for gname, gmask in groups.items():
        group_n[gname] = int(gmask.sum())
    for ps in psets:
        bold_te = stack(runs, run_field(runs, "bold_z", ps), tek)
        real_bold[ps] = {}
        for gname, gmask in groups.items():
            real_bold[ps][gname] = float(bold_te[gmask, 0].mean())
    res = dict(subject=sub, n_train_scans=n_tr, n_test_scans=n_te, n_train_runs=len(trk), n_test_runs=len(tek),
               parcel_sets=parcel_sets, parcel_overlap=overlap,
               t_map_corr=corr(t88, t_train), reference_r_danger_bold=ref, models={},
               group_n=group_n, real_bold_by_group=real_bold)
    for name in models:
        t1 = time.time()
        X, red, expl = model_features(runs, trk, ok_tr, name)
        Xte = stack(runs, X, tek)
        Wa, alpha_a = E.fit_episodes(*episodes(runs, trk, X, run_field(runs, "risk_z")))
        a_hat = (Xte @ Wa)[:, 0]
        out = dict(pca_explained=expl, alpha_danger=float(alpha_a[0]), r_danger=corr(a_hat, risk_te), sets={})
        # nulls for the danger readout: danger series circularly shifted inside each training run, same alpha
        null_a = shifted_nulls(runs, trk, X, Xte, Wa, alpha_a, rng, a.perm)
        if name in a.save_dirs:
            save_direction(a, sub, name, runs, trk, X, Wa, red)
        for ps in psets:
            out["sets"][ps] = brain_readout(runs, trk, tek, X, Xte, Wa, a_hat, null_a, groups, ps, a.perm)
        res["models"][name] = out
        s88 = out["sets"]["human88"]
        scl = out["sets"]["train_sel"]
        print(f"  {name:18s} danger r {out['r_danger']:+.3f} | brain r {s88['r_brain']:+.3f} "
              f"(clean {scl['r_brain']:+.3f}) | readouts r {s88['r_between_readouts']:+.3f} p {s88['p_between']:.3f} "
              f"(clean {scl['r_between_readouts']:+.3f}) | danger->BOLD {s88['r_danger_readout_to_bold']:+.3f} "
              f"p {s88['p_danger_to_bold']:.3f} | cos {s88['cos_whitened']:+.3f} ({time.time() - t1:.0f}s)", flush=True)
        os.makedirs(a.out, exist_ok=True)
        json.dump(res, open(f"{a.out}/{sub}.json", "w"), indent=1)
    real_txt = []
    for g_, v in res["real_bold_by_group"]["human88"].items():
        real_txt.append(f"{g_} {v:+.3f}")
    print("held-out means by threat type (z, per run): real BOLD human88 " + ", ".join(real_txt), flush=True)
    for name in models:
        bg = res["models"][name]["sets"]["human88"]["by_group"]
        cells = []
        for g_, v in bg.items():
            cells.append(f"{g_}: danger {v['danger_readout']:+.3f} brain {v['brain_readout']:+.3f}")
        print(f"  {name:18s} " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
