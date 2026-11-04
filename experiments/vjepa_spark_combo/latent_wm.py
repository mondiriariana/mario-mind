"""V-JEPA as a shared latent space, VLA-JEPA style (arXiv 2602.10098), with everything frozen except small heads.
The world model and the R^2 metric here are the ones glimmer_vlajepa/train.py uses on Glimmer.

Spark sees only the current frame (4x4 vision-tower tokens). K=4 learnable queries cross-attend to
those tokens and are squeezed to a small latent z (4 x 64; the "latent action"). A small world model takes the
current V-JEPA state s_t (2 s clip: block 47 mean + 2x2, and the newest tubelet 2x2) plus z, and predicts the
newest V-JEPA tubelet (block 47, 2x2) at +0.53, +1.07 and +2.0 s, as a residual on the current tubelet.
Future frames are only targets, never inputs (leakage-free, as in the paper).

Conditions:
  none        world model on s_t only (no latent)
  spark       s_t + z from Spark                        <- the VLA-JEPA analogue
  spark_shuf  s_t + z from Spark tokens of another frame (control: capacity / noise)
  vjepa_z     s_t + z from V-JEPA's own 4x4 tokens      (control: is it Spark, or just a second look)
  spark_only  z from Spark only, no s_t, no residual    (Spark mapped into V-JEPA space alone)
Metrics on held-out sessions (same 5 folds, inner session split for early stopping):
  r2_delta_h  variance-weighted R2 of the change (future - current tubelet); copying the present scores ~0
  r2_abs_h    same on the future tubelet itself
  probe_*     linear probes on z (frozen; balanced accuracy / R2), same probes as analyze.py --exp 1
Inputs in the frame set (--set): labels.csv, vjepa.npz (build_set.py), tokens_future.npz, tokens_spark.npz,
tokens_vjepa.npz (current tokens and future V-JEPA tubelets of the same frames).

  python latent_wm.py --conds all     # writes $MJG_ROOT/results/vjepa_spark_combo/exp5_latent_wm.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import OUT, SCRATCH, folds, load, log, means, probes, summarise  # noqa: E402

CONDS = ["none", "spark", "spark_shuf", "vjepa_z", "spark_only"]
K, DZ, D = 4, 64, 256


class Latent(nn.Module):
    """K learnable queries cross-attend to the (frozen) tokens of one encoder; output [B, K*DZ]."""
    def __init__(self, d_tok, n_tok=16):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_tok), nn.Linear(d_tok, D))
        self.pos = nn.Parameter(torch.randn(n_tok, D) * 0.02)
        self.q = nn.Parameter(torch.randn(K, D) * 0.02)
        self.att = nn.MultiheadAttention(D, 4, dropout=0.1, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, DZ))

    def forward(self, tok):
        h = self.proj(tok) + self.pos
        z, _ = self.att(self.q.expand(len(tok), K, D), h, h)
        return self.out(z).flatten(1)


class WM(nn.Module):
    def __init__(self, cond, d_s, d_tok, n_h, d_out, n_tok=16):
        super().__init__()
        self.cond, self.n_h, self.d_out = cond, n_h, d_out
        self.use_s = cond != "spark_only"
        self.use_z = cond != "none"
        d_in = 0
        if self.use_s:
            self.es = nn.Sequential(nn.LayerNorm(d_s), nn.Linear(d_s, 512), nn.GELU(), nn.Dropout(0.3))
            d_in += 512
        if self.use_z:
            self.lat = Latent(d_tok, n_tok)
            d_in += K * DZ
        self.mlp = nn.Sequential(nn.Linear(d_in, 1024), nn.GELU(), nn.Dropout(0.2), nn.Linear(1024, n_h * d_out))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s, tok, cur):
        parts, z = [], None
        if self.use_s:
            parts.append(self.es(s))
        if self.use_z:
            z = self.lat(tok)
            parts.append(z)
        y = self.mlp(torch.cat(parts, 1)).view(-1, self.n_h, self.d_out)
        if self.use_s:
            y = y + cur[:, None]  # residual on the current tubelet (copying the present is the starting point)
        return y, z


def r2_vw(y, p):
    """variance-weighted R2 over all dims."""
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=f"{SCRATCH}/vjepa_spark_combo/v1", help="frame set (build_set.py)")
    ap.add_argument("--conds", default="all")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-ep", type=int, default=60)
    ap.add_argument("--out", default=f"{OUT}/exp5_latent_wm.json")
    a = ap.parse_args()
    conds = CONDS if a.conds == "all" else a.conds.split(",")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    lab, classes, v, _ = load(a.set, need_spark=False)
    n = len(lab)
    fu = np.load(f"{a.set}/tokens_future.npz")
    horizons = fu["horizons"].tolist()
    cur = fu["cur_last"].astype(np.float32).reshape(n, -1)
    fut = fu["fut_last"].astype(np.float32).reshape(n, len(horizons), -1)
    valid = fu["valid"]
    S0 = np.hstack([v["depth_mean"][:, 3], v["b47_grid2"].reshape(n, -1), cur])
    tok = {"spark": np.load(f"{a.set}/tokens_spark.npz")["s4"].astype(np.float32)}
    if "vjepa_z" in conds:
        tok["vjepa"] = np.load(f"{a.set}/tokens_vjepa.npz")["v4"].astype(np.float32)
    groups = (lab["sub"] + "_" + lab["ses"]).values
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    horizons_s = []
    for h in horizons:
        horizons_s.append(round(h * 8 / 60, 2))
    res["_info"] = {"n": n, "horizons_rows": horizons, "horizons_s": horizons_s,
                    "valid_share": valid.mean(0).round(3).tolist(), "doc": __doc__.split("\n\n")[0]}
    for cond in conds:
        per, t0 = [], time.time()
        for fi, (tr0, te) in enumerate(folds(lab)[:a.folds]):
            torch.manual_seed(fi)
            rng = np.random.default_rng(fi)
            itr, iva = next(GroupShuffleSplit(1, test_size=0.15, random_state=fi).split(tr0, groups=groups[tr0]))
            tr, va = tr0[itr], tr0[iva]
            # standardise with train stats; the tubelet scale is shared by current and future (residual target)
            mu_s, sd_s = S0[tr].mean(0), S0[tr].std(0) + 1e-6
            mu_t, sd_t = cur[tr].mean(0), cur[tr].std(0) + 1e-6
            S = torch.tensor((S0 - mu_s) / sd_s)
            C = torch.tensor((cur - mu_t) / sd_t)
            Y = torch.tensor((fut - mu_t) / sd_t)
            Mk = torch.tensor(valid)
            src = tok["vjepa" if cond == "vjepa_z" else "spark"]
            mu_k, sd_k = src[tr].mean(0), src[tr].std(0) + 1e-6
            T = torch.tensor((src - mu_k) / sd_k)
            perm = np.arange(n)
            if cond == "spark_shuf":  # each frame gets the Spark tokens of a random frame (fixed per fold)
                perm = rng.permutation(n)
            T = T[perm]
            model = WM(cond, S.shape[1], T.shape[-1], len(horizons), C.shape[1])
            opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=0.05)

            def loss_on(idx):
                idx = torch.as_tensor(idx)
                p, _ = model(S[idx], T[idx], C[idx])
                m = Mk[idx].float()[..., None]
                return (((p - Y[idx]) ** 2) * m).sum() / (m.sum() * p.shape[-1]).clamp(min=1)

            best, state, bad = 1e9, None, 0
            for ep in range(a.max_ep):
                model.train()
                for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 256)):
                    l = loss_on(b)
                    opt.zero_grad()
                    l.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    losses = []
                    for b in np.array_split(va, max(1, len(va) // 1024)):
                        losses.append(float(loss_on(b)))
                    vl = float(np.mean(losses))
                if vl < best - 1e-4:
                    best, bad, state = vl, 0, {}
                    for k, t in model.state_dict().items():
                        state[k] = t.clone()
                else:
                    bad += 1
                    if bad >= 6:
                        break
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                P, Z = [], []
                for b in np.array_split(np.arange(n), max(1, n // 2048)):
                    p, z = model(S[b], T[b], C[b])
                    P.append(p)
                    Z.append(z)
                P = torch.cat(P).numpy()
                Z = torch.cat(Z).numpy() if Z[0] is not None else None
            m = {"epochs": ep + 1, "val_mse": best}
            Yn, Cn = Y.numpy(), C.numpy()
            for j, h in enumerate(horizons):
                ok = te[valid[te, j]]
                m[f"r2_delta_{h}"] = r2_vw(Yn[ok, j] - Cn[ok], P[ok, j] - Cn[ok])
                m[f"r2_abs_{h}"] = r2_vw(Yn[ok, j], P[ok, j])
            if Z is not None:
                pr = probes(Z[tr0], Z[te], lab.iloc[tr0], lab.iloc[te])
                for k, x in pr.items():
                    m[f"probe_{k}"] = x
            per.append(m)
        res[cond] = summarise(per)
        log(cond, f"({time.time() - t0:.0f}s)", means(res[cond]))
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
