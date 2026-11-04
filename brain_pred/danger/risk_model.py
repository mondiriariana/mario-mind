"""Danger experiment, step 2 (CPU): an objective danger scale = the chance that Mario dies in the next 3 seconds.

From the per-frame game state (frame_features.py), a gradient-boosted classifier predicts "a death within the next
--horizon seconds" from what is around Mario (nearest enemy ahead / behind and its height, number of enemies, nearest
hole ahead and its width, Mario's speed, height, airborne, size). Trained on the human plays of the non-held-out
sessions of all 5 subjects (configs/holdout_sessions.json), evaluated on the held-out sessions (AUC), then applied to
every play. Samples while Mario is already dying or fallen are left out.
Each sample also gets a threat type from what is closest ahead: "enemy" (enemy within --near px), "hole" (gap within
--near px), "both", or "none".
Writes results/danger/risk/<play>.npz (frame, risk, type), results/danger/risk_model.json (AUC, base rates) and
risk_model.pkl (the fitted classifier, to score the agent's games).

  $MJG_SCI_PY risk_model.py      (CPU node)
"""
import argparse, glob, json, os, re, time

import numpy as np

ROOT = os.environ.get("MJG_ROOT", ".")
CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
FR = f"{ROOT}/results/danger/frames"
OUT = f"{ROOT}/results/danger"
COLS = ["mx_screen", "my", "vx", "vy", "air", "big", "n_enemies", "enemy_ahead_dx", "enemy_ahead_dy", "enemy_behind_dx",
        "gap_ahead_dx", "gap_width"]


def feature_matrix(z):
    """[n, len(COLS)] float32: the risk model's inputs from a frame_features dict / npz."""
    cols = []
    for c in COLS:
        cols.append(z[c])
    return np.stack(cols, 1).astype(np.float32)


def load(pid, horizon_frames):
    z = np.load(f"{FR}/{pid}.npz")
    X = feature_matrix(z)
    f, df = z["frame"], z["death_frame"]
    y = np.zeros(len(f), bool)
    for d in df:
        y |= (f < d) & (f >= d - horizon_frames)
    ok = (z["dying"] == 0) & (z["fallen"] == 0)
    return X, y, ok, f, z


def threat_type(z, near):
    e = z["enemy_ahead_dx"] < near
    h = z["gap_ahead_dx"] < near
    return np.where(e & h, "both", np.where(e, "enemy", np.where(h, "hole", "none")))


def stack_usable(plays, i):
    """Concatenate item i (X or y) of every loaded play, on its usable samples only."""
    parts = []
    for x in plays:
        parts.append(x[i][x[2]])
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=float, default=3.0, help="seconds")
    ap.add_argument("--near", type=float, default=64, help="px: threat type")
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    hold = json.load(open(os.path.join(CODE, "configs", "holdout_sessions.json")))["holdout"]
    pids = []
    for p in glob.glob(f"{FR}/*.npz"):
        pids.append(os.path.basename(p)[:-4])
    pids.sort()
    train_pids, test_pids = [], []
    for p in pids:
        m = re.match(r"(sub-\d+)_(ses-\d+)_", p)
        if m.group(2) in set(hold.get(m.group(1), [])):
            test_pids.append(p)
        else:
            train_pids.append(p)
    hf = int(a.horizon * 60)
    t0 = time.time()
    tr = []
    for p in train_pids:
        tr.append(load(p, hf))
    te = []
    for p in test_pids:
        te.append(load(p, hf))
    Xtr, ytr, Xte, yte = stack_usable(tr, 0), stack_usable(tr, 1), stack_usable(te, 0), stack_usable(te, 1)
    print(f"train {len(Xtr)} samples ({ytr.mean():.3%} before a death), held-out {len(Xte)} ({yte.mean():.3%}) "
          f"({time.time() - t0:.0f}s)", flush=True)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=0)
    clf.fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    # how much each cue matters: AUC when one feature group is replaced by its "safe" value
    safe = dict(enemy_ahead_dx=999, enemy_behind_dx=999, gap_ahead_dx=999, gap_width=0, n_enemies=0, enemy_ahead_dy=0)
    abl = {}
    for name, cols in dict(enemies=["enemy_ahead_dx", "enemy_ahead_dy", "enemy_behind_dx", "n_enemies"],
                           holes=["gap_ahead_dx", "gap_width"]).items():
        X2 = Xte.copy()
        for c in cols:
            X2[:, COLS.index(c)] = safe[c]
        abl[name] = float(roc_auc_score(yte, clf.predict_proba(X2)[:, 1]))
    print(f"held-out AUC {auc:.3f}; without enemy cues {abl['enemies']:.3f}, without hole cues {abl['holes']:.3f}")
    os.makedirs(f"{OUT}/risk", exist_ok=True)
    import pickle
    pickle.dump(clf, open(f"{OUT}/risk_model.pkl", "wb"))  # scores the agent's games too (agent_response.py)
    for (X, y, ok, f, z), pid in zip(tr + te, train_pids + test_pids):
        risk = clf.predict_proba(X)[:, 1].astype(np.float32)
        risk[~ok] = np.nan
        np.savez_compressed(f"{OUT}/risk/{pid}.npz", frame=f, risk=risk, type=threat_type(z, a.near),
                            death_frame=z["death_frame"], death_cause=z["death_cause"])
    json.dump(dict(horizon_s=a.horizon, near_px=a.near, auc_heldout=auc, auc_without=abl, features=COLS,
                   train_samples=int(len(Xtr)), heldout_samples=int(len(Xte)), base_rate_train=float(ytr.mean()),
                   base_rate_heldout=float(yte.mean())), open(f"{OUT}/risk_model.json", "w"), indent=1)
    print(f"wrote {OUT}/risk/ ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
