"""Danger experiment: demo figure (CPU). Left: brain response vs danger level, one line per network (mean +- SE across
runs). Right: where the brain responds more to high danger than to safe moments (t map on the cortical surface).
Inputs: human_response.py outputs (or an agent's, agent_response.py: same format) in results/danger/.
Large fonts, few elements: made to be shown on a screen while talking.

  $MJG_PY plot_danger.py --subject sub-03 --out danger_sub-03.png [--theme dark]   (matplotlib, nilearn)
"""
import argparse, json, os, sys

import numpy as np

ROOT = os.environ.get("MJG_ROOT", ".")
OUT = f"{ROOT}/results/danger"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo"))
SHOW = [("SalVentAttn", "salience (\"something important!\")", "#e8553a"), ("Cont", "control (planning)", "#9b59d0"),
        ("Vis", "visual (what is on screen)", "#2bb673")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="sub-03")
    ap.add_argument("--src", default="human", help="human, or agent_<tag>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--t-min", type=float, default=3.0, help="brain map: show parcels with |t| above this")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import plotting
    from render import parcel_to_vertices
    r = json.load(open(f"{OUT}/{a.src}_{a.subject}.json"))
    t = np.load(f"{OUT}/{a.src}_{a.subject}_parcels.npz")["t_high_minus_safe"]
    bg, fg = ("#ffffff", "#222222") if a.theme == "light" else ("#121216", "#eeeeee")
    plt.rcParams.update({"font.size": 15, "text.color": fg, "axes.labelcolor": fg, "xtick.color": fg, "ytick.color": fg,
                         "axes.edgecolor": fg})
    fig = plt.figure(figsize=(16, 6.4), facecolor=bg)
    ax = fig.add_axes([0.08, 0.17, 0.36, 0.62], facecolor=bg)
    x = np.arange(4)
    for key, title, col in SHOW:
        j = r["networks"].index(key)
        m, s = [], []
        for resp_lv, se_lv in zip(r["response"], r["se"]):
            m.append(resp_lv[j])
            s.append(se_lv[j])
        m, s = np.array(m), np.array(s)
        star = " *" if r["p_perm"][j] < 0.05 else ""
        ax.errorbar(x, m, yerr=s, color=col, lw=3.5, marker="o", ms=9, capsize=4, label=title + star)
    ax.axhline(0, color=fg, lw=0.8, alpha=0.4)
    ticks = []
    for L, mr in zip(r["levels"], r["mean_risk_per_level"]):
        ticks.append(f"{L}\n{100 * mr:.0f}% risk")
    ax.set_xticks(x, ticks)
    ax.set_xlabel("danger (chance of dying in the next 3 s)")
    ax.set_ylabel("brain response (z)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=13, loc="upper left")
    who = "Human brain" if a.src == "human" else f"Agent ({a.src})"
    fig.text(0.08, 0.93, f"{who}: the more dangerous the moment, the stronger the alert response",
             fontsize=20, weight="bold", color=fg)
    fig.text(0.08, 0.875, f"{a.subject}, held-out sessions, close calls only (Mario survives); beyond buttons, "
             "brightness, motion and sound; * p < 0.05", fontsize=12, color=fg, alpha=0.7)
    fs, v2p = parcel_to_vertices()
    vmax = float(np.nanpercentile(np.abs(t), 99)) or 1.0
    for i, hemi in enumerate(("left", "right")):
        axb = fig.add_axes([0.48 + 0.26 * i, 0.12, 0.26, 0.72], projection="3d", facecolor=bg)
        lab = v2p[hemi]
        vals = np.where(lab > 0, t[np.clip(lab - 1, 0, len(t) - 1)], 0.0)
        plotting.plot_surf_stat_map(fs["inflated"].parts[hemi], vals, hemi=hemi, view="lateral", axes=axb,
                                    colorbar=False, vmax=vmax, threshold=a.t_min, cmap="cold_hot", bg_on_data=True,
                                    figure=fig)
        axb.set_facecolor(bg)
    n_up = int((t > a.t_min).sum())
    fig.text(0.74, 0.79, f"where: {n_up} regions respond more to high danger than to safe moments (in colour)",
             fontsize=13, color=fg, ha="center")
    fig.savefig(a.out, dpi=110, facecolor=bg)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
