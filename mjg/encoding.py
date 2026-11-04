"""Brain-encoding helpers for CNeuroMod Mario: play/scan timing and ridge regression.

Inputs live under $MJG_CNEUROMOD: data/mario.annotations (play onsets), data/mario.fmriprep (confounds, for
the scan counts) and the derived parcel time series derived/parcels/<sub>.h5 (Schaefer-1000; 6 motion + WM/CSF
+ cosine high-pass regressed out, detrended, z-scored per run), which are read, never written.

Timing: play video frame f is shown at onset + f / 60 s; replay screen k == video frame k (docs/DATA.md).
TR = 1.49 s, scan k covers [k TR, (k + 1) TR).
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CNEUROMOD = Path(os.environ.get("MJG_CNEUROMOD", os.path.join(os.environ.get("MJG_ROOT", "."), "cneuromod")))
DATA = CNEUROMOD / "data"
DERIVED = CNEUROMOD / "derived"
TR = 1.49
FPS = 60
TRIM = 5
DEFAULT_ALPHAS = np.logspace(-1, 6, 15)


# ----------------------------------------------------------------------------------------- timing
def run_key(ses: str, run: int) -> str:
    return f"{ses}_run-{run:02d}"


def plays(sub: str):
    """One row per play: sub, ses, run, key, onset, duration, level, phase, stem, n_frames."""
    import pandas as pd
    rows = []
    for f in sorted(glob.glob(str(DATA / "mario.annotations" / sub / "ses-*" / "func" / "*annotated_events.tsv"))):
        ses, run = re.search(r"(ses-\d+)_task-mario_run-(\d+)", f).groups()
        d = pd.read_csv(f, sep="\t")
        g = d[d.trial_type == "gym-retro_game"]
        for _, r in g.iterrows():
            rows.append(dict(sub=sub, ses=ses, run=int(run), key=run_key(ses, int(run)), onset=float(r.onset),
                             duration=float(r.duration), level=r.level, phase=r.phase,
                             stem=os.path.basename(r.stim_file).replace(".bk2", ""),
                             n_frames=int(r.frame_stop - r.frame_start)))
    return pd.DataFrame(rows)


def run_trs_from_confounds(sub: str) -> dict[str, int]:
    """Number of scans per run (rows of its fMRIPrep confounds file), keyed by run_key."""
    out = {}
    pattern = DATA / "mario.fmriprep" / sub / "ses-*" / "func" / "*desc-confounds_timeseries.tsv"
    for f in sorted(glob.glob(str(pattern))):
        ses, run = re.search(r"(ses-\d+)_task-mario_run-(\d+)", f).groups()
        with open(f) as fh:
            n_lines = 0
            for _ in fh:
                n_lines += 1
            out[run_key(ses, int(run))] = n_lines - 1
    return out


def scan_table(sub: str, n_trs: dict[str, int], p=None):
    """One row per scan: the play on screen (largest overlap), its frames [f0, f1) inside the scan window,
    coverage (share of the scan inside that play). Scans between plays: stem "" and coverage 0."""
    import pandas as pd
    p = plays(sub) if p is None else p
    rows = []
    for key, n in n_trs.items():
        pr = p[p.key == key].sort_values("onset")
        for k in range(n):
            t0, t1 = k * TR, (k + 1) * TR
            best, cov = None, 0.0
            for r in pr.itertuples():
                ov = min(t1, r.onset + r.duration) - max(t0, r.onset)
                if ov > cov:
                    best, cov = r, ov
            if best is None:
                rows.append(dict(key=key, tr=k, stem="", level="", f0=-1, f1=-1, coverage=0.0, t_play=-1.0))
                continue
            f0 = max(0, int(round((t0 - best.onset) * FPS)))
            f1 = min(best.n_frames, int(round((t1 - best.onset) * FPS)))
            rows.append(dict(key=key, tr=k, stem=best.stem, level=best.level, f0=f0, f1=f1, coverage=cov / TR,
                             t_play=max(0.0, t0 - best.onset)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------- ridge
def pearson_cols(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    den[den == 0] = np.inf
    return (a * b).sum(0) / den


def add_delays(X, delays=(1, 2, 3, 4, 5)):
    """FIR delay embedding within one run (delays in TRs)."""
    n, p = X.shape
    out = np.zeros((n, p * len(delays)), dtype=X.dtype)
    for k, d in enumerate(delays):
        out[d:, k * p:(k + 1) * p] = X[:n - d] if d > 0 else X
    return out


def _z(x):
    sd = x.std(0)
    sd[sd == 0] = 1
    return (x - x.mean(0)) / sd


@dataclass
class PCAReducer:
    """PCA fitted on training runs only (each run z-scored first)."""
    n_components: int = 256
    mean_: np.ndarray | None = field(default=None, repr=False)
    comps_: np.ndarray | None = field(default=None, repr=False)

    def fit(self, runs):
        zruns = []
        for r in runs:
            zruns.append(_z(r))
        X = np.concatenate(zruns).astype(np.float64)
        self.mean_ = X.mean(0)
        lam, V = np.linalg.eigh(np.cov(X - self.mean_, rowvar=False))
        order = np.argsort(lam)[::-1][: self.n_components]
        self.comps_ = V[:, order]
        self.explained_ = float(lam[order].sum() / lam.sum())
        return self

    def transform(self, run):
        return _z(((_z(run) - self.mean_) @ self.comps_).astype(np.float32))


def _stats(X, Y):
    X64 = X.astype(np.float64, copy=False)
    return X64.T @ X64, X64.T @ Y.astype(np.float64, copy=False), np.einsum("ij,ij->j", Y, Y, dtype=np.float64)


def _sse_all_alphas(Gtr, Ctr, Gva, Cva, sva, alphas):
    lam, V = np.linalg.eigh(Gtr)
    lam = np.clip(lam, 0, None)
    B, GvaV, CvaV = V.T @ Ctr, V.T @ Gva @ V, V.T @ Cva
    sse = np.empty((len(alphas), Ctr.shape[1]))
    for i, a in enumerate(alphas):
        Wt = B / (lam[:, None] + a)
        sse[i] = sva - 2 * np.einsum("pt,pt->t", Wt, CvaV) + np.einsum("pt,pt->t", Wt, GvaV @ Wt)
    return sse


def fit_episodes(X_eps, Y_eps, alphas=DEFAULT_ALPHAS, n_folds=5, seed=0):
    """Ridge with a per-target alpha chosen by episode-grouped CV (closed form from Gram matrices), then refit.
    X, Y z-scored within run beforehand (no intercept). Returns W (p, t), alphas (t,)."""
    alphas = np.asarray(alphas)
    order = np.random.default_rng(seed).permutation(len(X_eps))
    folds = np.array_split(order, min(n_folds, len(order)))
    fs = []
    for f in folds:
        X_fold = []
        for i in f:
            X_fold.append(X_eps[i])
        Y_fold = []
        for i in f:
            Y_fold.append(Y_eps[i])
        fs.append(_stats(np.concatenate(X_fold), np.concatenate(Y_fold)))
    G = 0
    for f in fs:
        G += f[0]
    C = 0
    for f in fs:
        C += f[1]
    sse = np.zeros((len(alphas), C.shape[1]))
    for Gf, Cf, sf in fs:
        sse += _sse_all_alphas(G - Gf, C - Cf, Gf, Cf, sf, alphas)
    best = alphas[sse.argmin(0)]
    lam, V = np.linalg.eigh(G)
    lam = np.clip(lam, 0, None)
    return V @ ((V.T @ C) / (lam[:, None] + best[None, :])), best


BAND_SCALES = (0.0, 0.1, 0.3, 1.0, 3.0)


def fit_episodes_banded(X_eps, Y_eps, band_mask, scales=BAND_SCALES, alphas=DEFAULT_ALPHAS, n_folds=5, seed=0):
    """Two-band ridge: the columns in band_mask (the feature band) are scaled by s before ridge, which gives them
    their own penalty (alpha / s^2) relative to the other columns (nuisance). (s, alpha) is chosen per target by
    episode-grouped CV; s = 0 drops the band, so a useless band costs ~nothing (the banded-ridge idea of
    Reason-to-Play / himalaya, restricted to two bands). Returns W (p, t) in the original column scale, alphas (t,),
    scales (t,)."""
    alphas, scales = np.asarray(alphas), np.asarray(scales, dtype=np.float64)
    order = np.random.default_rng(seed).permutation(len(X_eps))
    folds = np.array_split(order, min(n_folds, len(order)))
    fs = []
    for f in folds:
        X_fold = []
        for i in f:
            X_fold.append(X_eps[i])
        Y_fold = []
        for i in f:
            Y_fold.append(Y_eps[i])
        fs.append(_stats(np.concatenate(X_fold), np.concatenate(Y_fold)))
    G = 0
    for f in fs:
        G += f[0]
    C = 0
    for f in fs:
        C += f[1]
    sse = np.zeros((len(scales), len(alphas), C.shape[1]))
    for si, s in enumerate(scales):
        d = np.where(band_mask, s, 1.0)
        for Gf, Cf, sf in fs:
            Gs, Cs = d[:, None] * (G - Gf) * d[None, :], d[:, None] * (C - Cf)
            sse[si] += _sse_all_alphas(Gs, Cs, d[:, None] * Gf * d[None, :], d[:, None] * Cf, sf, alphas)
    flat = sse.reshape(-1, C.shape[1]).argmin(0)
    best_s, best_a = scales[flat // len(alphas)], alphas[flat % len(alphas)]
    W = np.zeros((G.shape[0], C.shape[1]))
    for si, s in enumerate(scales):
        cols = np.where(best_s == s)[0]
        if not len(cols):
            continue
        d = np.where(band_mask, s, 1.0)
        lam, V = np.linalg.eigh(d[:, None] * G * d[None, :])
        lam = np.clip(lam, 0, None)
        Ws = V @ ((V.T @ (d[:, None] * C[:, cols])) / (lam[:, None] + best_a[None, cols]))
        W[:, cols] = d[:, None] * Ws  # back to the original columns
    return W, best_a, best_s


MULTI_SCALES = (0.0, 0.3, 1.0, 3.0)


def fit_episodes_multiband(X_eps, Y_eps, band_ids, scales=MULTI_SCALES, alphas=DEFAULT_ALPHAS, n_folds=5, seed=0):
    """Banded ridge with one penalty per feature band: band_ids[c] = 0 for nuisance columns (scale 1),
    1..B for the feature bands. Every combination of per-band scales is tried (len(scales)^B), and (scales, alpha)
    is chosen per target by episode-grouped CV, as in fit_episodes_banded; a scale of 0 drops that band. So a
    small band (the agent) is not crowded out by a large one (V-JEPA) sharing its penalty.
    Returns W (p, t), alphas (t,), scales (t, B)."""
    import itertools
    band_ids = np.asarray(band_ids)
    B = int(band_ids.max())
    alphas = np.asarray(alphas)
    combos = np.array(list(itertools.product(scales, repeat=B)), dtype=np.float64)  # (n_combo, B)
    order = np.random.default_rng(seed).permutation(len(X_eps))
    folds = np.array_split(order, min(n_folds, len(order)))
    fs = []
    for f in folds:
        X_fold = []
        for i in f:
            X_fold.append(X_eps[i])
        Y_fold = []
        for i in f:
            Y_fold.append(Y_eps[i])
        fs.append(_stats(np.concatenate(X_fold), np.concatenate(Y_fold)))
    G = 0
    for f in fs:
        G += f[0]
    C = 0
    for f in fs:
        C += f[1]
    dcol = lambda c: np.concatenate([[1.0], c])[band_ids]
    sse = np.zeros((len(combos), len(alphas), C.shape[1]))
    for ci, c in enumerate(combos):
        d = dcol(c)
        for Gf, Cf, sf in fs:
            Gs, Cs = d[:, None] * (G - Gf) * d[None, :], d[:, None] * (C - Cf)
            sse[ci] += _sse_all_alphas(Gs, Cs, d[:, None] * Gf * d[None, :], d[:, None] * Cf, sf, alphas)
    flat = sse.reshape(-1, C.shape[1]).argmin(0)
    best_c, best_a = flat // len(alphas), alphas[flat % len(alphas)]
    W = np.zeros((G.shape[0], C.shape[1]))
    for ci in np.unique(best_c):
        cols = np.where(best_c == ci)[0]
        d = dcol(combos[ci])
        lam, V = np.linalg.eigh(d[:, None] * G * d[None, :])
        lam = np.clip(lam, 0, None)
        W[:, cols] = d[:, None] * (V @ ((V.T @ (d[:, None] * C[:, cols])) / (lam[:, None] + best_a[None, cols])))
    return W, best_a, combos[best_c]
