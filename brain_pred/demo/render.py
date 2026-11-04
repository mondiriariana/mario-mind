"""Demo video: the agent playing, its reasoning, its internal state and the predicted brain activity, in sync.

Inputs: a play folder (glimmer_play.py: steps.jsonl with each turn's reply, meta.json, inputs.npy), its replay
(brain_pred/generator/replay.py: frames + per-frame RAM), a fitted encoder (encode.py --save) for one subject,
optionally the agent's internal state per turn (hidden.npy [n_turns, d]). BOLD is predicted here with
brain_pred/features + prediction (the same steps as brain_pred/generator/generate_bold.py).
Panels (1280 x 720, 30 fps = every 2nd emulator frame):
  game (left)        the replayed screen, 2x
  reasoning (right)  perception + reasoning + answer of the turn being played
  brain (right)      predicted BOLD at the current TR on the fsaverage5 surface, left and right lateral views
                     (Schaefer-1000 parcels projected with the atlas), one image per TR
  networks (bottom)  predicted BOLD time course of four Schaefer networks (visual, somatomotor, dorsal attention,
                     control), with a cursor at the current time
  state (optional)   2-D PCA of the agent's internal state per turn, trajectory so far (same layer as the encoder's
                     agent band when there is one)
The BOLD is a model prediction (encoder held-out mean r is printed on the frame), not a measurement.
Highlighting (--highlight): colour only the regions where the encoder is trustworthy, grey elsewhere, and draw the
network curves from those regions only:
  all     every region (default)
  r-min   regions whose held-out r >= --r-min (e.g. 0.3)
  top-k   the --top-k regions with the highest held-out r
  beats   regions where this encoder's held-out r beats another encoder's (--vs <encoder.npz>) by > --margin

  python render.py --play <dir> --replay <replay.npz> --model <encoder.npz> --subject sub-03 --out demo.mp4
  (env $MJG_PY with nilearn, matplotlib, PyAV, Pillow; the atlas from $MJG_ATLAS)
"""
import argparse, json, os, re, sys, textwrap

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "features"))
sys.path.insert(0, os.path.join(HERE, "..", "prediction"))
ROOT = os.environ.get("MJG_ROOT", ".")
CNEUROMOD = os.environ.get("MJG_CNEUROMOD", os.path.join(ROOT, "cneuromod"))
ATLAS_DIR = os.environ.get("MJG_ATLAS", os.path.join(CNEUROMOD, "atlas"))
ATLAS = f"{ATLAS_DIR}/tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-1000Parcels7Networks_dseg.nii.gz"
LABELS = f"{ATLAS_DIR}/tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-1000Parcels7Networks_dseg.tsv"
NETWORKS = [("Vis", "visual"), ("SomMot", "somatomotor"), ("DorsAttn", "dorsal attention"), ("Cont", "control")]
W_, H_, FPS_OUT, STEP = 1280, 720, 30, 2


def turn_of_frame(play_dir, n_frames):
    """Emulator frame -> index of the turn being played (frames before the first turn: -1)."""
    meta = json.load(open(f"{play_dir}/meta.json"))
    steps = []
    for line in open(f"{play_dir}/steps.jsonl"):
        steps.append(json.loads(line))
    t_of = np.full(n_frames, -1, int)
    f = int(meta.get("start_frames", 0))
    for i, s in enumerate(steps):
        held = int(s.get("frames_held", 1))
        n = held + (1 if held > 1 else 0)  # hold + one NOOP frame
        t_of[f:f + n] = i
        f += n
    t_of[f:] = len(steps) - 1
    return t_of, steps


def parcel_to_vertices():
    """fsaverage5 vertex -> Schaefer parcel id (0 = none), per hemisphere, via the volumetric atlas."""
    from nilearn import datasets, surface
    fs = datasets.load_fsaverage("fsaverage5")
    out = {}
    for hemi in ("left", "right"):
        lab = surface.vol_to_surf(ATLAS, fs["pial"].parts[hemi], interpolation="nearest_most_frequent",
                                  radius=3.0, n_samples=10)
        out[hemi] = np.rint(lab).astype(int)
    return fs, out


def highlight_mask(r, how, r_min=0.3, top_k=200, r_vs=None, margin=0.0):
    """Parcels to colour (bool [n_parcels]) from the encoder's held-out r per parcel; see --highlight."""
    r = np.nan_to_num(np.asarray(r, np.float64), nan=-1)
    if how == "r-min":
        return r >= r_min
    if how == "top-k":
        m = np.zeros(len(r), bool)
        m[np.argsort(-r)[:top_k]] = True
        return m
    if how == "beats":
        return r - np.nan_to_num(np.asarray(r_vs, np.float64), nan=-1) > margin
    return np.ones(len(r), bool)


def brain_images(y, fs, v2p, vmax, size=(560, 260), mask=None):
    """One image per TR: left and right lateral surfaces coloured by predicted BOLD (only parcels in mask)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import plotting
    from PIL import Image
    imgs = []
    for t in range(len(y)):
        fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(size[0] / 100, size[1] / 100), dpi=100,
                                 facecolor=(18 / 255, 18 / 255, 22 / 255))
        for ax, hemi in zip(axes, ("left", "right")):
            lab = v2p[hemi]
            idx = np.clip(lab - 1, 0, y.shape[1] - 1)
            keep = (lab > 0) & (mask[idx] if mask is not None else True)
            vals = np.where(keep, y[t][idx], 0.0)
            plotting.plot_surf_stat_map(fs["inflated"].parts[hemi], vals, hemi=hemi, view="lateral", axes=ax,
                                        colorbar=False, vmax=vmax, cmap="cold_hot",
                                        threshold=vmax * 0.15, figure=fig)
        for ax in axes:
            ax.set_facecolor((18 / 255, 18 / 255, 22 / 255))
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        fig.canvas.draw()
        rgba = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
        imgs.append(np.asarray(rgba.convert("RGB")))
        plt.close(fig)
    return imgs


def fit_reply(rep, draw, font, max_h, width=70, spacing=3):
    """Wrap a turn's reply to at most max_h pixels; if it is too long, cut it with '…' and keep the answer line."""
    text = " ".join(rep.split())
    lines = textwrap.wrap(text, width)
    pitch = draw.textbbox((0, 0), "A", font=font)[3] + spacing  # line pitch of multiline_text
    n_max = max(1, (max_h - draw.textbbox((0, 0), "Ag[]", font=font)[3]) // pitch + 1)
    if len(lines) <= n_max:
        return "\n".join(lines)
    m = re.search(r"<answer>.*?</answer>", text)
    ans = textwrap.wrap(m.group(0), width)[: n_max - 1] if m else []
    keep = lines[: n_max - len(ans)]
    keep[-1] = keep[-1][: width - 2].rstrip() + " …"
    return "\n".join(keep + ans)


def network_means(y, mask=None):
    import pandas as pd
    lab = pd.read_csv(LABELS, sep="\t")
    names = lab["name"].astype(str).values if "name" in lab else lab.iloc[:, 1].astype(str).values
    out = {}
    for key, title in NETWORKS:
        cols = []
        for i, n in enumerate(names):
            if f"_{key}_" in n and i < y.shape[1] and (mask is None or mask[i]):
                cols.append(i)
        out[title] = y[:, cols].mean(1) if cols else np.zeros(len(y))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--subject", default="sub-03")
    ap.add_argument("--hidden", default=None, help="agent internal state per turn, [n_turns, d] .npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--highlight", choices=["all", "r-min", "top-k", "beats"], default="all")
    ap.add_argument("--r-min", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--vs", default=None, help="--highlight beats: the other encoder (.npz from encode.py --save)")
    ap.add_argument("--margin", type=float, default=0.0)
    a = ap.parse_args()
    from PIL import Image, ImageDraw, ImageFont
    import av
    from play_features import tr_features, turn_starts, agent_tr_features
    from predict import EncoderModel

    rp = np.load(a.replay)
    frames, ram, names, level = rp["frames"], rp["ram"], list(rp["button_names"]), str(rp["level"])
    inputs = np.load(f"{a.play}/inputs.npy")
    gen = EncoderModel(a.model)
    raw = tr_features(inputs, ram[1:], names, level,          # RAM after each input, aligned with the inputs
                      lowlevel=rp["lowlevel"] if "lowlevel" in rp else None)
    if os.path.exists(f"{a.play}/hidden.npy"):                 # agent_* encoders: BOLD from the agent's own state
        raw["agent"] = agent_tr_features(np.load(f"{a.play}/hidden.npy"), turn_starts(a.play), len(raw["buttons"]))
    y = gen.predict(raw)                                       # [n_TR, 1000]
    vmax = float(np.percentile(np.abs(y[5:] if len(y) > 6 else y), 98)) or 1.0
    r_vs = EncoderModel(a.vs).r_heldout if a.vs else None
    mask = highlight_mask(gen.r_heldout, a.highlight, a.r_min, a.top_k, r_vs, a.margin)
    vs_name = os.path.basename(a.vs).replace(".npz", "") if a.vs else ""
    hl_text = {"all": "",
               "r-min": f"colour: the {mask.sum()} regions this encoder predicts with held-out r >= {a.r_min}",
               "top-k": f"colour: the {mask.sum()} regions this encoder predicts best (held-out r)",
               "beats": f"colour: the {mask.sum()} regions where it beats {vs_name}"}[a.highlight]
    fs, v2p = parcel_to_vertices()
    brains = brain_images(y, fs, v2p, vmax, mask=None if a.highlight == "all" else mask)
    nets = network_means(y, None if a.highlight == "all" else mask)
    agent_depth = []
    for b in gen.bands:
        if b.startswith("agent_"):
            agent_depth.append(int(b.rsplit("_d", 1)[1]))
    t_of, steps = turn_of_frame(a.play, len(frames))
    hid2d = None
    if a.hidden:
        h = np.load(a.hidden).astype(np.float32)
        depth = agent_depth[0] if agent_depth else 2  # the encoder's layer, else layer 39 of 52
        if h.ndim == 3:  # glimmer_play --save-hidden: [n_turns, 4 depths, hidden] (layers 13 / 26 / 39 / 52)
            h = h[:, depth]
        h = h - h.mean(0)
        _, _, Vt = np.linalg.svd(h, full_matrices=False)
        hid2d = h @ Vt[:2].T
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = big = ImageFont.load_default()

    out = av.open(a.out, "w")
    st = out.add_stream("libx264", rate=FPS_OUT)
    st.width, st.height, st.pix_fmt = W_, H_, "yuv420p"
    tr_len = 1.49 * 60
    r_mean = float(np.nanmean(gen.r_heldout))
    for fi in range(0, len(frames), STEP):
        canvas = Image.new("RGB", (W_, H_), (18, 18, 22))
        d = ImageDraw.Draw(canvas)
        g = Image.fromarray(frames[fi]).resize((frames.shape[2] * 2, frames.shape[1] * 2), Image.NEAREST)
        canvas.paste(g, (10, 40))
        d.text((10, 10), f"Agent playing {level}  |  frame {fi}  |  t = {fi / 60:5.1f} s", font=big,
               fill=(240, 240, 240))
        ti = t_of[min(fi, len(t_of) - 1)]
        if ti >= 0:
            rep = steps[ti].get("reply") or ""
            txt = fit_reply(rep, d, font, max_h=292 - 66 - 8)  # stop above the brain heading at y = 292
            d.text((W_ // 2 + 20, 40), f"Turn {ti + 1}: reasoning", font=big, fill=(255, 210, 120))
            d.multiline_text((W_ // 2 + 20, 66), txt, font=font, fill=(220, 220, 220), spacing=3)
        k = min(int(fi // tr_len), len(y) - 1)
        if k >= 0:
            d.text((W_ // 2 + 20, 292), f"Predicted brain activity ({a.subject}), TR {k}", font=big,
                   fill=(180, 200, 255))
            d.text((W_ // 2 + 20, 316), f"model prediction, not a measurement: encoder held-out mean r {r_mean:.2f}",
                   font=font, fill=(150, 150, 165))
            if hl_text:
                d.text((W_ // 2 + 20, 336), hl_text + "; grey: not shown", font=font, fill=(150, 150, 165))
            canvas.paste(Image.fromarray(brains[k]), (W_ // 2 + 40, 352 if hl_text else 340))
        # network time courses
        x0, y0, w, h = 10, 560, W_ // 2 - 20, 150
        d.rectangle([x0, y0, x0 + w, y0 + h], outline=(70, 70, 80))
        cols = [(120, 200, 255), (255, 160, 90), (140, 230, 140), (230, 120, 220)]
        span = 1e-6
        for v in nets.values():
            span = max(span, np.abs(v).max())
        for (title, v), c, j in zip(nets.items(), cols, range(4)):
            pts = []
            for i in range(len(v)):
                pts.append((x0 + w * i / max(1, len(v) - 1), y0 + h / 2 - (h / 2.4) * v[i] / span))
            if len(pts) > 1:
                d.line(pts, fill=c, width=2)
            d.text((x0 + 6 + 140 * j, y0 + 4), title, font=font, fill=c)
        cx = x0 + w * min(fi / tr_len, len(y) - 1) / max(1, len(y) - 1)
        d.line([(cx, y0), (cx, y0 + h)], fill=(250, 250, 250), width=1)
        if hid2d is not None and ti >= 0:
            p = hid2d[: ti + 1]
            lo, hi = hid2d.min(0), hid2d.max(0) + 1e-6
            bx0, by0, bw, bh = W_ // 2 + 20, 560, W_ // 2 - 30, 150  # bottom right, next to the network panel
            d.rectangle([bx0, by0, bx0 + bw, by0 + bh], outline=(70, 70, 80))
            q = []
            for u, v in p:
                q.append((bx0 + 10 + (bw - 20) * (u - lo[0]) / (hi[0] - lo[0]),
                          by0 + bh - 10 - (bh - 30) * (v - lo[1]) / (hi[1] - lo[1])))
            d.text((bx0 + 6, by0 + 4), f"Agent's internal state (PCA of layer {13 * (depth + 1)}, one point per turn)",
                   font=font, fill=(200, 200, 200))
            if len(q) > 1:
                d.line(q, fill=(200, 200, 120), width=1)
            d.ellipse([q[-1][0] - 4, q[-1][1] - 4, q[-1][0] + 4, q[-1][1] + 4], fill=(255, 80, 80))
        for pkt in st.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")):
            out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()
    print(f"{a.out}: {len(range(0, len(frames), STEP))} frames at {FPS_OUT} fps, {len(y)} TRs of predicted BOLD")


if __name__ == "__main__":
    main()
