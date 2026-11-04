"""Danger steering, analysis (CPU): what changes in the SFT model's play and writing when its danger direction is added.

For every strength folder results/danger/steer/<kind>_a<alpha>/<scenario>/run-000 (steer_play.py / job_steer.sbatch),
per scenario game, from the agent's own play only (frames after the human prefix):
  play     died (meta end == death), seconds played, progress (x), mean risk (risk_model.pkl on the RAM at each turn
           start, dying / fallen left out: the same danger scale as the human analysis), A presses per second (from
           every frame's inputs), distance (px) to the nearest gap / enemy ahead at turns where it chooses to jump
           (larger = jumps earlier), mean horizontal speed at turn starts
  buttons  share of turns pressing a (jump), b (run), left, noop
  writing  per 100 reasoning words: the trace word families (brain_pred/agent/trace_features.py), format errors,
           tokens per reply
  state    projection of the saved layer state on the steering direction (manipulation check: rises with alpha)
Each strength is compared with alpha 0 on the same scenarios (paired difference, mean and bootstrap 95% CI over
scenarios). Writes results/danger/steer/summary_<prefix><kind>.json and prints a table.

  $MJG_SCI_PY steer_analyze.py --kind pattern --dir <danger direction npz>       (the risk model's sklearn)
"""
import argparse, glob, json, os, pickle, re, sys

import numpy as np

ROOT = os.environ.get("MJG_ROOT", ".")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "agent"))
import frame_features as FF  # noqa: E402
from risk_model import feature_matrix  # noqa: E402
from trace_features import FAMILIES  # noqa: E402

OUT = f"{ROOT}/results/danger"
WRITE_FAM = ["danger", "caution", "enemy", "gap", "jump", "uncertain", "speed"]
BUTTONS = ("a", "b", "left", "noop")
SHOW = ["died", "seconds", "progress", "mean_risk", "jumps_per_s", "jump_dist", "speed", "btn_a", "btn_b", "btn_left",
        "btn_noop", "format_errors", "tokens"]


def family_patterns():
    rx = {}
    for k, v in FAMILIES.items():
        rx[k] = re.compile(rf"\b(?:{v})", re.I)
    return rx


RX = family_patterns()


def read_steps(d):
    steps, bad = [], 0
    for line in open(f"{d}/steps.jsonl"):
        try:
            steps.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"  {d}: {bad} unreadable line(s) in steps.jsonl skipped", flush=True)
    return steps


def game(d, clf, u, depth):
    """Play, buttons, writing and state of one game. ram.npy is the RAM at the start of each turn (+ the end),
    inputs.npy every frame's buttons (human prefix first)."""
    meta = json.load(open(f"{d}/meta.json"))
    start = int(meta.get("prefix_frames", meta.get("start_frames", 0)))
    ram = np.load(f"{d}/ram.npy")
    fz = FF.features(ram, 1)  # one row per turn start
    risk = clf.predict_proba(feature_matrix(fz))[:, 1]
    alive = (fz["dying"] == 0) & (fz["fallen"] == 0)
    thr = np.minimum(fz["gap_ahead_dx"], fz["enemy_ahead_dx"])
    secs = max(1e-6, (int(meta.get("frames", start)) - start) / 60)
    inp = np.load(f"{d}/inputs.npy")[start:]
    A = inp[:, list(meta["button_names"]).index("A")] if len(inp) else np.zeros(0, bool)
    a_on = int((A & ~np.r_[False, A[:-1]]).sum()) if len(A) else 0
    steps = read_steps(d)
    btn, texts, errors, tokens = [], [], [], []
    for s in steps:
        btn.append(s.get("buttons") or [])
        texts.append(s.get("reasoning") or "")
        errors.append(s.get("error") is not None)
        tokens.append(s.get("new_tokens", 0))
    jt = []
    for i, b in enumerate(btn):
        if "a" in b and i < len(thr) and thr[i] < 200:
            jt.append(i)
    rea = " ".join(texts)
    nw = max(1, len(rea.split()))
    out = dict(died=float(meta.get("end") == "death"), seconds=secs, progress=float(meta.get("progress", 0)),
               mean_risk=float(np.nanmean(risk[alive])) if alive.any() else np.nan,
               jumps_per_s=a_on / secs, jump_dist=float(np.mean(thr[jt])) if jt else np.nan,
               speed=float(np.mean(fz["vx"][alive])) if alive.any() else np.nan,
               turns=len(steps), format_errors=float(np.mean(errors)) if steps else np.nan,
               tokens=float(np.mean(tokens)) if steps else np.nan)
    for b in BUTTONS:
        pressed = []
        for x in btn:
            pressed.append(b in x)
        out[f"btn_{b}"] = float(np.mean(pressed)) if btn else np.nan
    for fam in WRITE_FAM:
        out[f"w_{fam}"] = 100 * len(RX[fam].findall(rea)) / nw
    hp = f"{d}/hidden.npy"
    if u is not None and os.path.exists(hp):
        out["state_proj"] = float((np.load(hp)[:, depth].astype(np.float32) @ u).mean())
    return out


def compare(G, al, keys, rng, boot):
    """Means at strength al and paired differences vs alpha 0 (bootstrap 95% CI over scenarios)."""
    common = sorted(set(G[al]) & set(G[0.0]))
    n_enemy, n_gap = 0, 0
    for c in common:
        n_enemy += c.startswith("enemy")
        n_gap += c.startswith("hole")
    row = dict(n_games=len(common), n_enemy=n_enemy, n_gap=n_gap, means={}, diff_vs_0={})
    for k in keys:
        v, v0 = [], []
        for c in common:
            v.append(G[al][c].get(k, np.nan))
            v0.append(G[0.0][c].get(k, np.nan))
        v, v0 = np.array(v), np.array(v0)
        row["means"][k] = float(np.nanmean(v)) if np.isfinite(v).any() else None
        dd = (v - v0)[np.isfinite(v - v0)]
        if al != 0 and len(dd) > 2:
            bs = []
            for _ in range(boot):
                bs.append(rng.choice(dd, len(dd)).mean())
            row["diff_vs_0"][k] = dict(mean=float(dd.mean()), lo=float(np.percentile(bs, 2.5)),
                                       hi=float(np.percentile(bs, 97.5)), n=len(dd))
    return row


def print_table(res, als, show):
    header = []
    for al in als:
        header.append(f"{f'alpha {al:+g}':>34s}")
    print(f"{'':16s}" + " | ".join(header))
    for k in show:
        cells = []
        for al in als:
            r = res["alphas"][str(al)]
            m = r["means"].get(k)
            dv = r["diff_vs_0"].get(k)
            cell = f"{m:.3f}" if m is not None else "-"
            if dv:
                cell += f" ({dv['mean']:+.3f} [{dv['lo']:+.3f},{dv['hi']:+.3f}])"
            cells.append(f"{cell:>34s}")
        print(f"{k:16s}" + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", default="pattern")
    ap.add_argument("--dir", default=f"{OUT}/direction/sub-03_agent_sft_d1_danger_dir.npz")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--prefix", default="", help="output-folder prefix of job_steer.sbatch (OUTTAG), e.g. <tag>_")
    a = ap.parse_args()
    clf = pickle.load(open(f"{OUT}/risk_model.pkl", "rb"))
    z = np.load(a.dir) if os.path.exists(a.dir) else None
    u, depth = (z[a.kind].astype(np.float32), int(z["depth"])) if z is not None else (None, 1)
    folders = {}
    for p in glob.glob(f"{OUT}/steer/{a.prefix}{a.kind}_a*"):
        folders[float(re.search(r"_a(-?[\d.]+)$", p).group(1))] = p
    G = {}
    for al, p in sorted(folders.items()):
        G[al] = {}
        for d in sorted(glob.glob(f"{p}/*/run-000")):
            if os.path.exists(f"{d}/meta.json"):
                G[al][os.path.basename(os.path.dirname(d))] = game(d, clf, u, depth)
        print(f"alpha {al:+g}: {len(G[al])} games", flush=True)
    if 0.0 not in G:
        raise SystemExit("no alpha 0 folder: nothing to compare with")
    keys = list(next(iter(G[0.0].values())).keys())
    rng = np.random.default_rng(0)
    res = dict(kind=a.kind, direction=a.dir, alphas={})
    for al in sorted(G):
        res["alphas"][str(al)] = compare(G, al, keys, rng, a.boot)
    os.makedirs(f"{OUT}/steer", exist_ok=True)
    json.dump(res, open(f"{OUT}/steer/summary_{a.prefix}{a.kind}.json", "w"), indent=1)
    show = list(SHOW)
    for fam in WRITE_FAM:
        show.append(f"w_{fam}")
    if u is not None:
        show.append("state_proj")
    print_table(res, sorted(G), show)


if __name__ == "__main__":
    main()
