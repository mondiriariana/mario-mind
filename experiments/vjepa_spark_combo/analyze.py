"""Analyses of the V-JEPA + Spark frame set (build_set.py), and the loaders, session folds and probes that
glimmer_vlajepa/train.py and latent_wm.py reuse. Needs numpy, pandas, scikit-learn (and torch for --exp 2).

All splits: 5 folds grouped by (subject, session), so test frames always come from sessions never seen
in training. Held-out fMRI sessions are not in the set at all.

  --exp 1  complementarity: linear probes (game state, next action) on V-JEPA, Spark vision, and both
  --exp 2  action prediction: small MLP from V-JEPA, Spark vision, or both, to the human's next action
  --exp 3  V-JEPA -> Spark language model: ridge maps to every LM layer (R^2 on held-out sessions), and
           linear probes on LM layers, on the `lm_subset` frames
These also read the Spark features of the same frames (spark_vision.npz, spark_lm/batch_*.npz in the set).
Writes $MJG_ROOT/results/vjepa_spark_combo/exp<N>_*.json.

  python analyze.py --exp 1 --set <frame set dir>
"""
import argparse, glob, json, os, time
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

BIN = ["airborne", "enemy_ahead", "enemy_near", "pit_ahead", "wall_ahead", "block_in_reach"]
CONT = ["x_screen", "y_pos"]
ROOT = os.environ.get("MJG_ROOT", ".")
SCRATCH = os.environ.get("MJG_SCRATCH", os.path.join(ROOT, "scratch"))
OUT = f"{ROOT}/results/vjepa_spark_combo"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def as_float32(arrays):
    out = {}
    for k, x in arrays.items():
        out[k] = x.astype(np.float32)
    return out


def load(set_dir, need_spark=True):
    """labels (with the action class index `cls`), class names, V-JEPA arrays, Spark vision arrays."""
    lab = pd.read_csv(f"{set_dir}/labels.csv")
    for c in BIN + ["lm_subset"]:
        lab[c] = lab[c].astype(str).str.lower().eq("true")
    classes = sorted(lab["human_class"].astype(str).unique())
    index = {}
    for i, c in enumerate(classes):
        index[c] = i
    lab["cls"] = lab["human_class"].astype(str).map(index)
    v = dict(np.load(f"{set_dir}/vjepa.npz"))
    s = dict(np.load(f"{set_dir}/spark_vision.npz")) if need_spark else {}
    return lab, classes, as_float32(v), as_float32(s)


def folds(lab, k=5):
    g = (lab["sub"] + "_" + lab["ses"]).values
    return list(GroupKFold(k).split(np.zeros(len(lab)), groups=g))


def reduce(blocks, tr, te, k):
    """Each block: standardise + PCA(k) fitted on train only; blocks concatenated."""
    A, B = [], []
    for X in blocks:
        sc = StandardScaler().fit(X[tr])
        p = PCA(min(k, len(tr) - 1), random_state=0).fit(sc.transform(X[tr]))
        A.append(p.transform(sc.transform(X[tr])))
        B.append(p.transform(sc.transform(X[te])))
    return np.hstack(A), np.hstack(B)


def probes(A, B, ltr, lte):
    """Linear probes trained on A (labels ltr), scored on B (labels lte): balanced accuracy / R^2 per label."""
    out = {}
    sa = StandardScaler().fit(A)
    A, B = sa.transform(A), sa.transform(B)
    for c in BIN + ["cls"]:
        if ltr[c].nunique() < 2 or lte[c].nunique() < 2:
            continue
        m = LogisticRegression(C=0.05, max_iter=1000, class_weight="balanced").fit(A, ltr[c])
        out["next_action" if c == "cls" else c] = balanced_accuracy_score(lte[c], m.predict(B))
    for c in CONT:
        out[c] = r2_score(lte[c], RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(A, ltr[c]).predict(B))
    return out


def summarise(per_fold):
    """{metric: [mean, sd]} over the folds."""
    out = {}
    for k in per_fold[0].keys():
        vals = []
        for f in per_fold:
            vals.append(f[k])
        out[k] = [round(float(np.mean(vals)), 3), round(float(np.std(vals)), 3)]
    return out


def means(summary):
    """{metric: mean} of a summarise() result, for the log."""
    out = {}
    for k, v in summary.items():
        out[k] = v[0]
    return out


def four_depths(depth_mean):
    """The 4 V-JEPA depths of a [n, 4, d] array as 4 blocks of [n, d]."""
    blocks = []
    for d in range(4):
        blocks.append(depth_mean[:, d])
    return blocks


def exp1(a):
    lab, classes, v, s = load(a.set)
    sets = {  # name -> (blocks, PCA dims per block)
        "vjepa_b47 (256)": ([v["depth_mean"][:, 3]], 256),
        "vjepa_b47 (512)": ([v["depth_mean"][:, 3]], 512),
        "vjepa_4depths (4x128)": (four_depths(v["depth_mean"]), 128),
        "spark_tower (256)": ([s["tower_mean"]], 256),
        "spark_tower (512)": ([s["tower_mean"]], 512),
        "spark_perception (256)": ([s["perc_mean"]], 256),
        "vjepa_b47 + spark_tower (256+256)": ([v["depth_mean"][:, 3], s["tower_mean"]], 256),
        "vjepa_4depths + spark_tower (4x128+256)": (four_depths(v["depth_mean"]) + [s["tower_mean"]], None),
    }
    F = folds(lab)
    res = {"n": len(lab), "classes": classes,
           "metric": "balanced accuracy (binary / next_action), R2 (x_screen, y_pos); mean, sd over 5 session folds"}
    for name, (blocks, k) in sets.items():
        per = []
        for tr, te in F:
            if k is None:
                A1, B1 = reduce(blocks[:4], tr, te, 128)
                A2, B2 = reduce(blocks[4:], tr, te, 256)
                A, B = np.hstack([A1, A2]), np.hstack([B1, B2])
            else:
                A, B = reduce(blocks, tr, te, k)
            per.append(probes(A, B, lab.iloc[tr], lab.iloc[te]))
        res[name] = summarise(per)
        log(name, means(res[name]))
    json.dump(res, open(f"{OUT}/exp1_probes.json", "w"), indent=1)


def exp2(a):
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lab, classes, v, s = load(a.set)
    n = len(lab)
    V = np.hstack([v["depth_mean"][:, 3], v["b47_grid2"].reshape(n, -1)])
    S = np.hstack([s["tower_mean"], s["tower_grid2"].reshape(n, -1)])
    sets = {"vjepa (b47 mean + 2x2)": [V], "spark_tower (mean + 2x2)": [S], "vjepa + spark_tower": [V, S]}
    y = lab["cls"].values
    res = {"n": n, "classes": classes, "target": "human_class: the human's action class over the next 0.25 s",
           "model": "LayerNorm -> Linear(512) -> GELU -> Dropout 0.3 -> Linear(n_classes); AdamW 1e-3, wd 0.05, "
                    "15 epochs, batch 256, class-weighted cross-entropy; no tuning on test folds",
           "chance_balanced": round(1 / len(classes), 3)}
    for name, blocks in sets.items():
        per = []
        for fi, (tr, te) in enumerate(folds(lab)):
            X = np.hstack(blocks)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xtr = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32, device=dev)
            Xte = torch.tensor((X[te] - mu) / sd, dtype=torch.float32, device=dev)
            ytr = torch.tensor(y[tr], device=dev)
            torch.manual_seed(fi)
            net = nn.Sequential(nn.LayerNorm(X.shape[1]), nn.Linear(X.shape[1], 512), nn.GELU(), nn.Dropout(0.3),
                                nn.Linear(512, len(classes))).to(dev)
            w = torch.tensor(len(tr) / (len(classes) * np.bincount(y[tr], minlength=len(classes)).clip(1)),
                             dtype=torch.float32, device=dev)
            opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.05)
            for ep in range(15):
                net.train()
                for b in torch.randperm(len(tr), device=dev).split(256):
                    loss = nn.functional.cross_entropy(net(Xtr[b]), ytr[b], weight=w)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                pred = net(Xte).argmax(1).cpu().numpy()
            m = {"balanced_acc": balanced_accuracy_score(y[te], pred), "acc": float((pred == y[te]).mean())}
            for i, c in enumerate(classes):
                m[f"recall_{c}"] = float((pred[y[te] == i] == i).mean())
            per.append(m)
        res[name] = summarise(per)
        log(name, means(res[name]))
    json.dump(res, open(f"{OUT}/exp2_action.json", "w"), indent=1)


def rounded(xs, nd):
    out = []
    for x in xs:
        out.append(round(x, nd))
    return out


def load_spark_lm(set_dir):
    """Frame indices and Spark language-model states ({"img", "last"}: [n, layers, 8192]) of spark_lm/batch_*.npz."""
    parts = []
    for f in sorted(glob.glob(f"{set_dir}/spark_lm/batch_*.npz")):
        parts.append(dict(np.load(f)))
    idx_parts = []
    for p in parts:
        idx_parts.append(p["idx"])
    lm = {}
    for k in ("img", "last"):
        arrays = []
        for p in parts:
            arrays.append(p[k])
        lm[k] = np.concatenate(arrays).astype(np.float32)
    return np.concatenate(idx_parts), lm


def layer_r2(Y, reduced, F):
    """Mean over the folds of the variance-weighted R^2 of a ridge map from the reduced inputs to Y."""
    scores = []
    for (A, B), (tr, te) in zip(reduced, F):
        p = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(A, Y[tr]).predict(B)
        scores.append(r2_score(Y[te], p, multioutput="variance_weighted"))
    return float(np.mean(scores))


def exp3(a):
    lab, classes, v, s = load(a.set)
    idx, lm = load_spark_lm(a.set)
    sub = lab.iloc[idx].reset_index(drop=True)
    n_layers = lm["img"].shape[1]
    log(f"{len(idx)} frames with Spark LM states, {n_layers} layers")
    preds = {"vjepa_b47": [v["depth_mean"][idx, 3]], "vjepa_4depths": four_depths(v["depth_mean"][idx]),
             "spark_tower": [s["tower_mean"][idx]], "spark_perception": [s["perc_mean"][idx]],
             "vjepa_b47 + spark_tower": [v["depth_mean"][idx, 3], s["tower_mean"][idx]]}
    k_of = {"vjepa_4depths": 64}
    F = folds(sub)
    res = {"n": len(idx), "layers": int(n_layers),
           "metric": "variance-weighted R2 of the 8192-d layer state on held-out sessions (mean over 5 folds)",
           "map": {}, "probes": {}}
    red = {}
    for name, bl in preds.items():
        red[name] = []
        for tr, te in F:
            red[name].append(reduce(bl, tr, te, k_of.get(name, 128)))
    for kind in ("img", "last"):
        for name in preds:
            r2 = []
            for L in range(lm[kind].shape[1]):
                r2.append(layer_r2(lm[kind][:, L], red[name], F))
            res["map"][f"{kind} | {name}"] = rounded(r2, 3)
            log(kind, name, "R2 by layer (every 10th):", rounded(r2[::10], 2), "last", round(r2[-1], 2))
    layers = sorted(set(list(range(0, n_layers, 6)) + [n_layers - 1]))
    probe_sets = {}
    for L in layers:
        probe_sets[f"lm_img_L{L}"] = [lm["img"][:, L]]
    for L in layers[::2]:
        probe_sets[f"lm_last_L{L}"] = [lm["last"][:, L]]
    probe_sets.update({"vjepa_b47": preds["vjepa_b47"], "spark_tower": preds["spark_tower"]})
    for name, bl in probe_sets.items():
        per = []
        for tr, te in F:
            per.append(probes(*reduce(bl, tr, te, 128), sub.iloc[tr], sub.iloc[te]))
        res["probes"][name] = summarise(per)
        log("probe", name, means(res["probes"][name]))
    json.dump(res, open(f"{OUT}/exp3_lm.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--set", default=f"{SCRATCH}/vjepa_spark_combo/v1", help="frame set (build_set.py)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {1: exp1, 2: exp2, 3: exp3}[a.exp](a)
