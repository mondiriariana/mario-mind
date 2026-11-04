"""Danger experiment (CPU): the agent's brain response to danger, one sample per decision (turn).

The scan-level analysis (agent_response.py) keeps few scans for agents that die soon after taking over in a danger
scenario, while the slow BOLD read-out needs 4.5-7.5 s after each scan. Here every decision of the agent in its own
scenario games is one sample:
  - danger = peak risk (same risk model and the human's level thresholds) over the 0.5 s after the decision;
  - close calls only: no death within 1.5 s of the decision; frames where Mario is dying / fallen are left out;
  - brain pattern of the decision = the encoder's response to that internal state held over the delays (the agent
    band's weights summed over the 5 delays; nuisance columns ignored), i.e. which brain pattern the encoder ties to
    that state, without the hemodynamic timing.
Then the same summaries as the human analysis: response per danger level and network, high - safe with a permutation
test (levels shuffled within each game), per-parcel t map and its similarity to the human danger map, plus the
correlation of danger (continuous) with each network's value. The timing differs from the human BOLD analysis.
Uses the games and prep files of agent_response.py --stage prep (results/danger/agent_<tag>/, agent_<tag>_prep/).

  $MJG_SCI_PY agent_decisions.py --tag <tag> --subject sub-03
Writes results/danger/agent_<tag>_<sub>_decisions.json (+ _parcels.npz), same keys as the scan-level file.
"""
import argparse, glob, json, os, pickle, sys

import numpy as np

ROOT = os.environ.get("MJG_ROOT", ".")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "features"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from play_features import FPS, turn_starts  # noqa: E402
from risk_model import feature_matrix, threat_type  # noqa: E402
from human_response import NETS, network_means, parcel_networks, type_counts  # noqa: E402
from agent_response import best_depth, level_response  # noqa: E402
from mjg import encoding as E  # noqa: E402

OUT = f"{ROOT}/results/danger"
WINDOW, DEATH_GAP = int(0.5 * FPS), int(1.5 * FPS)  # frames


def agent_weights(path):
    """-> (W_sum [d_pca, n_parcels], pca mean, pca comps, band): the agent band's weights summed over the delays."""
    z = np.load(path)
    bands = []
    for x in z["bands"]:
        bands.append(str(x))
    n_delays, W = len(z["delays"]), z["W"]
    b = bands[-1]
    assert b.startswith("agent_") and f"pca_comps__{b}" in z, bands
    m, C = z[f"pca_mean__{b}"], z[f"pca_comps__{b}"]
    p, w = W.shape[0] // n_delays, C.shape[1]
    off = p - w  # the agent band is the last one in each delay block (see predict.EncoderModel.design)
    W_sum = 0
    for k in range(n_delays):
        W_sum = W_sum + W[k * p + off:k * p + off + w]
    return W_sum, m, C, b


def ranks(v):
    return np.argsort(np.argsort(v)).astype(float)


def rank_corr(v, netY):
    """Spearman correlation of v with each network's value."""
    out = []
    for k in range(len(NETS)):
        out.append(np.corrcoef(ranks(v), ranks(netY[:, k]))[0, 1])
    return np.array(out)


def collect_samples(a, clf):
    """Per usable decision: the layer state and (game, peak risk, threat type)."""
    H, rows = [], []
    for pf in sorted(glob.glob(f"{OUT}/agent_{a.tag}_prep/*.npz")):
        name = os.path.basename(pf)[:-4]
        d = f"{OUT}/agent_{a.tag}/{name}/run-000"
        fz = dict(np.load(pf))
        hid = np.load(f"{d}/hidden.npy", mmap_mode="r")
        starts = turn_starts(d)[:len(hid)]
        risk = clf.predict_proba(feature_matrix(fz))[:, 1]
        risk[(fz["dying"] > 0) | (fz["fallen"] > 0)] = np.nan
        typ, deaths = threat_type(fz, 64), fz["deaths"]
        for i, s in enumerate(starts):
            w = (fz["frame"] >= s) & (fz["frame"] < s + WINDOW) & np.isfinite(risk)
            if not w.any() or ((deaths >= s) & (deaths < s + DEATH_GAP)).any():
                continue
            j = np.where(w)[0][np.nanargmax(risk[w])]
            H.append(np.asarray(hid[i, a.depth], np.float32))
            rows.append((name, float(risk[j]), str(typ[j])))
    return H, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--subject", default="sub-03")
    ap.add_argument("--depth", type=int, default=-1, help="-1 = the layer whose encoder has the best held-out mean r")
    ap.add_argument("--perm", type=int, default=1000)
    a = ap.parse_args()
    if a.depth < 0:
        a.depth = best_depth(a.subject, a.tag)
    Wsum, m, C, band = agent_weights(f"{ROOT}/results/bold_generator/{a.subject}/agent_{a.tag}_d{a.depth}.npz")
    clf = pickle.load(open(f"{OUT}/risk_model.pkl", "rb"))
    human = json.load(open(f"{OUT}/human_{a.subject}.json"))
    q = np.array(human["risk_thresholds"])
    net = parcel_networks()

    H, rows = collect_samples(a, clf)
    if not rows:
        sys.exit(f"{a.tag}: no usable decision")
    game, risk, typ = [], [], []
    for row_game, row_risk, row_type in rows:
        game.append(row_game)
        risk.append(row_risk)
        typ.append(row_type)
    game, risk, typ = np.array(game), np.array(risk), np.array(typ)
    # z-score over all of this agent's decision states, then the encoder's PCA (as EncoderModel.design)
    x = E._z(np.stack(H))
    x = E._z(((x - m) @ C).astype(np.float32))
    Y = x @ Wsum  # [n_decision, n_parcels]
    lvl = np.digitize(risk, q)
    netY = network_means(Y, net)

    resp, se = level_response(netY, lvl, game)
    has = []
    for L in range(4):
        has.append((lvl == L).any())
    diff = netY[lvl == 3].mean(0) - netY[lvl == 0].mean(0) if has[0] and has[3] else np.full(7, np.nan)
    rng = np.random.default_rng(0)
    rho_obs, null_d, null_r = rank_corr(risk, netY), [], []
    for _ in range(a.perm):
        perm = np.arange(len(lvl))
        for g in np.unique(game):
            idx = np.where(game == g)[0]
            perm[idx] = rng.permutation(idx)
        lp = lvl[perm]
        null_d.append(netY[lp == 3].mean(0) - netY[lp == 0].mean(0) if has[0] and has[3] else np.zeros(7))
        null_r.append(rank_corr(risk[perm], netY))
    p_d = ((np.abs(np.array(null_d)) >= np.abs(diff)).sum(0) + 1) / (a.perm + 1)
    p_r = ((np.abs(np.array(null_r)) >= np.abs(rho_obs)).sum(0) + 1) / (a.perm + 1)
    hi, lo = Y[lvl == 3], Y[lvl == 0]
    t_parcel = (hi.mean(0) - lo.mean(0)) / np.sqrt(hi.var(0) / max(1, len(hi)) + lo.var(0) / max(1, len(lo)) + 1e-12) \
        if has[0] and has[3] else np.full(Y.shape[1], np.nan)
    hd = np.array(human["high_minus_safe"])
    hp = np.load(f"{OUT}/human_{a.subject}_parcels.npz")["t_high_minus_safe"]
    sim = float(np.corrcoef(diff, hd)[0, 1]) if np.isfinite(diff).all() else float("nan")
    sim_p = float(np.corrcoef(t_parcel, hp)[0, 1]) if np.isfinite(t_parcel).all() else float("nan")
    n_per_level, mean_risk = [], []
    for L in range(4):
        n_per_level.append(int((lvl == L).sum()))
        mean_risk.append(float(risk[lvl == L].mean()) if has[L] else None)
    res = dict(subject=a.subject, agent=a.tag, depth=a.depth, encoder=band, readout="decision",
               n_games=int(len(np.unique(game))), n_decisions=int(len(rows)),
               n_per_level=n_per_level, risk_thresholds=q.tolist(), mean_risk_per_level=mean_risk,
               networks=NETS, levels=human["levels"], response=resp, se=se, high_minus_safe=diff.tolist(),
               p_perm=p_d.tolist(), risk_rank_corr=rho_obs.tolist(), p_risk_corr=p_r.tolist(),
               similarity_to_human_networks=sim, similarity_to_human_parcels=sim_p,
               type_counts=type_counts(typ),
               note="one sample per decision of the agent (close calls); brain pattern = encoder response to the "
                    "internal state summed over delays, agent band only; no hemodynamic timing (unlike the human data)")
    out = f"{OUT}/agent_{a.tag}_{a.subject}_decisions"
    json.dump(res, open(f"{out}.json", "w"), indent=1)
    np.savez_compressed(f"{out}_parcels.npz", t_high_minus_safe=t_parcel.astype(np.float32))
    print(f"agent {a.tag}: {res['n_games']} games, {len(rows)} decision samples, per level {res['n_per_level']}")
    for k, n in enumerate(NETS):
        cells = []
        for L in range(4):
            cells.append(f"{resp[L][k]:+.3f}")
        print(f"{n:12s} " + "  ".join(cells) + f"   high-safe {diff[k]:+.3f} p {p_d[k]:.3f}"
              f"   danger rank-r {rho_obs[k]:+.3f} p {p_r[k]:.3f}   (human {hd[k]:+.3f})")
    print(f"similarity to the human danger pattern: networks r = {sim:.2f}, parcels r = {sim_p:.2f}")


if __name__ == "__main__":
    main()
