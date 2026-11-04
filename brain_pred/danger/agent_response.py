"""Danger experiment, step 3c (CPU): the agent's generated brain response to danger, in its own scenario games.

For every scenario game of one agent (job_scenarios.sbatch: results/danger/agent_<tag>/<scenario>/run-000):
  1. replay it (generator/replay.py: every frame + RAM + low-level confounds) and score its danger with the same risk
     model as the humans (risk_model.pkl on frame_features.features of its own frames);
  2. generate its brain with the subject's encoder for this agent
     (results/bold_generator/<sub>/agent_<tag>_d<depth>.npz),
     keeping ONLY the part that comes from the agent's internal state (the nuisance columns -- buttons, level, time,
     brightness, motion, sound -- are set to zero), so what the brain shows comes from its "mind", not its hands;
  3. same analysis as human_response.py: per scan of the agent's own play (after it took over), peak risk, the
     human's danger thresholds, close calls only (no death within 5 scans), response = generated signal at t+3 / t+4.
Writes results/danger/agent_<tag>_<sub>.json (+ _parcels.npz), same format as the human file (plot_danger.py
--src agent_<tag>), plus the similarity of its network pattern (high - safe) to the human's.
Two stages, because the emulator and the risk model live in different environments:
  $MJG_PY     agent_response.py --stage prep    --tag sft --subject sub-03 --depth 1   (stable-retro)
  $MJG_SCI_PY agent_response.py --stage analyze --tag sft --subject sub-03 --depth 1   (the risk model's sklearn)
prep writes results/danger/agent_<tag>_prep/<scenario>.npz (game features per 0.1 s, deaths, start, generated signal).
"""
import argparse, glob, json, os, pickle, sys

import numpy as np

ROOT = os.environ.get("MJG_ROOT", ".")
CNEUROMOD = os.environ.get("MJG_CNEUROMOD", os.path.join(ROOT, "cneuromod"))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("features", "prediction", "generator"):
    sys.path.insert(0, os.path.join(HERE, "..", p))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from play_features import TR, FPS  # noqa: E402
import frame_features as FF  # noqa: E402
from risk_model import feature_matrix, threat_type  # noqa: E402
from human_response import NETS, network_means, parcel_networks, type_counts  # noqa: E402
from mjg import encoding as E  # noqa: E402

OUT = f"{ROOT}/results/danger"


def best_depth(subject, tag):
    """Index (0-3) of the agent layer whose brain encoder has the best held-out mean r."""
    r = []
    for d in range(4):
        r.append(np.nanmean(np.load(f"{ROOT}/results/brain_encoding/{subject}_lowlevel/agent_{tag}_d{d}.npz")["r"]))
    depth = int(np.argmax(r))
    print(f"best layer for {tag}: index {depth} (held-out mean r {np.round(r, 4).tolist()})", flush=True)
    return depth


def level_response(netY, lvl, group):
    """Mean network response per danger level (mean over games, SE across games); NaN for an empty level."""
    resp, se = [], []
    for L in range(4):
        per_game = []
        for g in np.unique(group):
            sel = (lvl == L) & (group == g)
            if sel.any():
                per_game.append(netY[sel].mean(0))
        pr = np.array(per_game)
        resp.append(pr.mean(0).tolist() if len(pr) else [np.nan] * 7)
        se.append((pr.std(0) / np.sqrt(max(1, len(pr)))).tolist() if len(pr) else [np.nan] * 7)
    return resp, se


def agent_only_prediction(enc, f):
    """Generated BOLD from the agent band alone: the encoder's design with every non-agent column set to zero."""
    parts, is_agent = [], []
    for b in enc.bands:
        x = E._z(enc.band(f, b).astype(np.float32))
        if b in enc.pca:
            m, C = enc.pca[b]
            x = E._z(((x - m) @ C).astype(np.float32))
        parts.append(x)
        is_agent += [b.startswith("agent_")] * x.shape[1]
    X = E.add_delays(np.hstack(parts).astype(np.float32), enc.delays)
    keep = np.tile(np.array(is_agent), len(enc.delays))
    return (X * keep[None]) @ enc.W, X @ enc.W


def prep(a):
    from replay import replay
    from play_features import tr_features, agent_tr_features, turn_starts
    from predict import EncoderModel
    enc = EncoderModel(f"{ROOT}/results/bold_generator/{a.subject}/agent_{a.tag}_d{a.depth}.npz")
    od = f"{OUT}/agent_{a.tag}_prep"
    os.makedirs(od, exist_ok=True)
    for d in sorted(glob.glob(f"{OUT}/agent_{a.tag}/*/run-000")):
        name = os.path.basename(os.path.dirname(d))
        finished = os.path.exists(f"{d}/meta.json") and os.path.exists(f"{d}/hidden.npy")
        if not finished or os.path.exists(f"{od}/{name}.npz"):
            continue
        meta = json.load(open(f"{d}/meta.json"))
        _, ram, _, low = replay(d, a.integration)
        ram_after = ram[1:]  # RAM after each input = the human files' convention
        f = tr_features(np.load(f"{d}/inputs.npy"), ram_after, meta["button_names"], meta["level"], lowlevel=low)
        f["agent"] = agent_tr_features(np.load(f"{d}/hidden.npy"), turn_starts(d), len(f["buttons"]))
        y_agent, y_full = agent_only_prediction(enc, f)
        fz = FF.features(ram_after, 6)
        death_frames = []
        for x in FF.deaths(ram_after):
            death_frames.append(x[0])
        np.savez_compressed(f"{od}/{name}.npz", y_agent=y_agent.astype(np.float32), y_full=y_full.astype(np.float32),
                            deaths=np.array(death_frames, int),
                            start=int(meta.get("prefix_frames", meta.get("start_frames", 0))), **fz)
        print(f"prep {name}: {len(f['buttons'])} scans", flush=True)


def collect_scans(a, clf):
    """Scans of the agent's own play with their peak risk / threat type and generated response at t+3, t+4."""
    rows, Ys, scen_of = [], [], []
    for pf in sorted(glob.glob(f"{OUT}/agent_{a.tag}_prep/*.npz")):
        fz = dict(np.load(pf))
        y_agent, deaths, start = fz["y_agent"], fz["deaths"], int(fz["start"])
        T = len(y_agent)
        risk = clf.predict_proba(feature_matrix(fz))[:, 1]
        risk[(fz["dying"] > 0) | (fz["fallen"] > 0)] = np.nan
        typ = threat_type(fz, 64)
        for t in range(E.TRIM, T - 4):
            f0, f1 = int(round(t * TR * FPS)), int(round((t + 1) * TR * FPS))
            if f0 < start:  # only the agent's own play
                continue
            m = (fz["frame"] >= f0) & (fz["frame"] < f1) & np.isfinite(risk)
            if not m.any() or ((deaths >= f0) & (deaths < f0 + 5 * TR * FPS)).any():
                continue
            i = np.where(m)[0][np.nanargmax(risk[m])]
            rows.append(dict(risk=float(risk[i]), type=str(typ[i])))
            Ys.append(y_agent[t + 3:t + 5].mean(0))
            scen_of.append(os.path.basename(pf)[:-4])
    return rows, Ys, scen_of


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["prep", "analyze"], required=True)
    ap.add_argument("--tag", required=True, help="agent: sft / plain (folder results/danger/agent_<tag>)")
    ap.add_argument("--subject", default="sub-03")
    ap.add_argument("--depth", type=int, default=-1,
                    help="agent layer index 0-3 (Glimmer: decoder layers 13 / 26 / 39 / 52); -1 = the layer whose "
                         "encoder has the best held-out mean r")
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--integration",
                    default=os.environ.get("MJG_INTEGRATION", os.path.join(CNEUROMOD, "mario.stimuli")))
    a = ap.parse_args()
    if a.depth < 0:
        a.depth = best_depth(a.subject, a.tag)
    if a.stage == "prep":
        return prep(a)
    clf = pickle.load(open(f"{OUT}/risk_model.pkl", "rb"))
    human = json.load(open(f"{OUT}/human_{a.subject}.json"))
    q = np.array(human["risk_thresholds"])
    net = parcel_networks()
    rows, Ys, scen_of = collect_scans(a, clf)
    risk, typ = [], []
    for row in rows:
        risk.append(row["risk"])
        typ.append(row["type"])
    risk, typ = np.array(risk), np.array(typ)
    Yr = np.stack(Ys)
    scen = np.array(scen_of)
    lvl = np.digitize(risk, q)
    netY = network_means(Yr, net)
    resp, se = level_response(netY, lvl, scen)
    diff = netY[lvl == 3].mean(0) - netY[lvl == 0].mean(0)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(a.perm):
        lp = lvl.copy()
        for s in np.unique(scen):
            idx = np.where(scen == s)[0]
            lp[idx] = rng.permutation(lp[idx])
        null.append(netY[lp == 3].mean(0) - netY[lp == 0].mean(0))
    p = ((np.abs(np.array(null)) >= np.abs(diff)).sum(0) + 1) / (a.perm + 1)
    hi, lo = Yr[lvl == 3], Yr[lvl == 0]
    t_parcel = (hi.mean(0) - lo.mean(0)) / np.sqrt(hi.var(0) / max(1, len(hi)) + lo.var(0) / max(1, len(lo)) + 1e-12)
    hd = np.array(human["high_minus_safe"])
    sim = float(np.corrcoef(diff, hd)[0, 1])  # same pattern across the 7 networks as the human's?
    hp = np.load(f"{OUT}/human_{a.subject}_parcels.npz")["t_high_minus_safe"]
    sim_parcels = float(np.corrcoef(t_parcel, hp)[0, 1])
    n_per_level, mean_risk = [], []
    for L in range(4):
        n_per_level.append(int((lvl == L).sum()))
        mean_risk.append(float(risk[lvl == L].mean()) if (lvl == L).any() else None)
    res = dict(subject=a.subject, agent=a.tag, depth=a.depth, encoder=f"agent_{a.tag}_d{a.depth}",
               n_games=len(np.unique(scen)), n_scans=int(len(rows)), n_per_level=n_per_level,
               risk_thresholds=q.tolist(), mean_risk_per_level=mean_risk,
               networks=NETS, levels=human["levels"], response=resp, se=se, high_minus_safe=diff.tolist(),
               p_perm=p.tolist(),
               similarity_to_human_networks=sim, similarity_to_human_parcels=sim_parcels,
               type_counts=type_counts(typ),
               note="generated from the agent's internal state only (nuisance columns zeroed)")
    json.dump(res, open(f"{OUT}/agent_{a.tag}_{a.subject}.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/agent_{a.tag}_{a.subject}_parcels.npz", t_high_minus_safe=t_parcel.astype(np.float32))
    print(f"agent {a.tag}: {res['n_games']} games, {len(rows)} scans, per level {res['n_per_level']}")
    for j, n in enumerate(NETS):
        cells = []
        for L in range(4):
            cells.append(f"{resp[L][j]:+.4f}")
        print(f"{n:12s} " + "  ".join(cells) + f"   high-safe {diff[j]:+.4f}  p {p[j]:.3f}"
              f"   (human {hd[j]:+.3f})")
    print(f"similarity to the human danger pattern: networks r = {sim:.2f}, parcels r = {sim_parcels:.2f}")


if __name__ == "__main__":
    main()
