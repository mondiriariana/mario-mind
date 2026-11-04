"""Prediction: apply a fitted encoder to per-TR features -> predicted BOLD (Schaefer-1000 parcels, one row per TR).

An encoder is one file written by brain_pred/encoding/encode.py --save, for one subject and one model (feature set):
weights W [p, 1000], the bands in order ("nuis" first), the level list of the nuisance band, the FIR delays
(1-5 TRs), the PCA of each wide band (fitted on training runs), and its held-out-session accuracy (r per parcel).
Preprocessing is the training one: each band z-scored over the play (training z-scores per run), wide bands through
the stored PCA, FIR delays, then X @ W. Output units: the training BOLD (z-scored parcel signal). The first 5 TRs
have incomplete delays (trimmed in training).

Features come from brain_pred/features (new plays) or brain_pred/encoding/build_tr_features.py (human runs).

  from predict import EncoderModel
  bold = EncoderModel("<encoder.npz>").predict(features)   # [n_TR, 1000]
"""
import os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mjg import encoding as E  # noqa: E402

DEPTHS = (11, 23, 37, 47)  # V-JEPA 2.1 G blocks stored in the per-TR "vjepa" band


class EncoderModel:
    """encoder = EncoderModel(path); bold = encoder.predict(features)  # [n_TR, n_parcels]"""

    def __init__(self, path):
        z = np.load(path)
        self.path = path
        self.W = z["W"]
        self.bands, self.levels, delays = [], [], []
        for b in z["bands"]:
            self.bands.append(str(b))
        for l in z["levels"]:
            self.levels.append(str(l))
        for d in z["delays"]:
            delays.append(int(d))
        self.delays = tuple(delays)
        self.r_heldout = z["r_heldout"]
        self.nuis_lowlevel = bool(z["nuis_lowlevel"]) if "nuis_lowlevel" in z else False  # encode.py --lowlevel
        self.pca = {}
        for b in self.bands:
            if f"pca_mean__{b}" in z:
                self.pca[b] = (z[f"pca_mean__{b}"], z[f"pca_comps__{b}"])

    def _has_band(self, prefix):
        for b in self.bands:
            if b.startswith(prefix):
                return True
        return False

    @property
    def needs_vjepa(self):
        return self._has_band("vjepa")

    @property
    def needs_agent(self):
        return self._has_band("agent_")

    def band(self, f, b):
        T = len(f["buttons"])
        if b == "nuis":
            level = np.asarray(f["level"])
            one_hot = []
            for l in self.levels:
                one_hot.append(level == l)
            lv = np.stack(one_hot, 1).astype(np.float32)
            if self.nuis_lowlevel and "lowlevel" not in f:
                raise KeyError("this encoder's nuisance band has the low-level confounds: pass the play's `lowlevel`")
            parts = [f["buttons"], lv, f["t_play"][:, None] / 300]
            if self.nuis_lowlevel:
                parts.append(f["lowlevel"])
            return np.hstack(parts)
        if b == "gamestate":
            return f["gamestate"]
        if b == "vjepa4":
            return np.asarray(f["vjepa"], np.float32).reshape(T, -1)
        if b.startswith("vjepa") and b[5:].isdigit():
            return np.asarray(f["vjepa"], np.float32)[:, DEPTHS.index(int(b[5:]))]
        if b.startswith("agent_"):  # agent_<tag>_d<depth>: the agent's internal state at that depth
            return np.asarray(f["agent"], np.float32)[:, int(b.rsplit("_d", 1)[1])]
        raise KeyError(f"band {b} cannot be computed for a new play")

    def design(self, f):
        parts = []
        for b in self.bands:
            x = E._z(self.band(f, b).astype(np.float32))
            if b in self.pca:
                m, C = self.pca[b]
                x = E._z(((x - m) @ C).astype(np.float32))
            parts.append(x)
        return E.add_delays(np.hstack(parts).astype(np.float32), self.delays)

    def predict(self, f):
        return self.design(f) @ self.W
