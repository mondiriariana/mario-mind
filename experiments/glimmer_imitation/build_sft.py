"""SFT data for Glimmer: human plays cut into turns with the agent's own interface.

Per play (one subject, default sub-01; fMRI held-out sessions of configs/holdout_sessions.json never used;
validation = every 6th remaining session, offset 2; train = the rest):
  1. replay the .bk2 (buttons + RAM per frame)
  2. cut turns from the first frame Mario is in control (player_state 8): at a turn start t the label is the
     14-combo of the human's buttons at frame t; that combo implies the Odysseus hold (30 frames with A,
     10 with a direction, else 1, plus one frame after a hold > 1), so the next turn starts at
     t + hold (+1). The human may change buttons inside the hold: the label is what they pressed at t.
     Stops at the first death (as an agent episode) or at the end of the play.
  3. observation of a turn = the frames of the previous hold (every --every-th + the last; the first turn:
     its start frame), exactly as glimmer_play.py builds them
  4. replay again keeping only those frames
Writes <out>/<split>/<play_id>.npz (frames uint8 [m, 224, 240, 3]; obs_start/obs_len [n_turns] index the
frames of each turn's observation; combo [n_turns]; t [n_turns] turn start frames) and <out>/index.jsonl.

  python build_sft.py --subject sub-01 --out <dir>/sub-01      # CPU, emulator env ($MJG_PY)
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "..")
sys.path.insert(0, CODE)
ROOT = os.environ.get("MJG_ROOT", ".")
SCRATCH = os.environ.get("MJG_SCRATCH", os.path.join(ROOT, "scratch"))
CNEUROMOD = os.environ.get("MJG_CNEUROMOD", os.path.join(ROOT, "cneuromod"))
DATA = os.environ.get("MJG_DATA", f"{CNEUROMOD}/data/mario")
INTEG = os.environ.get("MJG_INTEGRATION", f"{CNEUROMOD}/mario.stimuli")
OUT = f"{SCRATCH}/glimmer_sft/v1"
DIRS = {"LEFT", "RIGHT", "UP", "DOWN"}


def hold_of(combo):
    from mjg.actions import COMBOS
    c = set(COMBOS[combo])
    return 30 if "A" in c else 10 if c & DIRS else 1


def cut_turns(buttons, names, ram, every, all_lives=False, lives_out=None):
    """-> list of (t, combo, obs frame indices). all_lives: instead of stopping at the first death, skip the death
    and go on from the next frame Mario is in control, through every life of the play; lives_out (a list) then
    gets each turn's life index (0 = first)."""
    from mjg.actions import ENV_BUTTONS, buttons_to_action
    col = {}
    for i, name in enumerate(names):
        col[name] = i
    pressed = np.zeros((len(buttons), len(ENV_BUTTONS)), bool)
    for j, name in enumerate(ENV_BUTTONS):
        if name and name in col:
            pressed[:, j] = buttons[:, col[name]]
    combos, _ = buttons_to_action(pressed)
    state, lives = ram["player_state"], ram["lives"]
    ctrl = np.where(state == 8)[0]
    if not len(ctrl):
        return []
    t, n = int(ctrl[0]), len(buttons)
    turns, obs = [], [t]
    life_ref, life = lives[ctrl[0]], 0
    while t < n:
        if state[t] in (6, 11) or lives[t] < life_ref:
            if not all_lives:
                break
            d = t
            while d < n and state[d] != 8:  # the death, then until Mario is in control again
                d += 1
            if d >= n:
                break
            t, life_ref, obs = d, lives[d], [d]
            life += 1
            continue
        c = int(combos[t])
        turns.append((t, c, obs))
        if lives_out is not None:
            lives_out.append(life)
        h = hold_of(c)
        end = min(n, t + h + (1 if h > 1 else 0))
        obs = []
        for i, f in enumerate(range(t, min(n, t + h))):
            if i % every == every - 1:
                obs.append(f)
        obs.append(end - 1)
        t = end
    return turns


def work(job):
    """job = (play dict, split, out folder, every[, all_lives]) -> index row; writes <out>/<split>/<play_id>.npz."""
    play, split, out, every = job[:4]
    all_lives = len(job) > 4 and job[4]  # brain_pred/agent/build_scan_turns.py --all-lives
    from mjg import emulator as E
    dst = f"{out}/{split}/{play['play_id']}.npz"
    if os.path.exists(dst):
        return dict(play, split=split, skipped=True)
    r1 = E.replay_bk2(play["bk2"], INTEG, keep_frame=lambda k: False, ram_vars=["player_state", "lives"])
    life = []
    turns = cut_turns(r1["buttons"], r1["button_names"], r1["ram"], every, all_lives, life)
    if not turns:
        return dict(play, split=split, n_turns=0)
    need = set()
    for _, _, o in turns:
        need.update(o)
    r2 = E.replay_bk2(play["bk2"], INTEG, keep_frame=lambda k: k in need, ram_vars=[])
    pos = {}
    for i, k in enumerate(r2["frame_idx"]):
        pos[int(k)] = i
    starts, lens, flat, combos, times = [], [], [], [], []
    for t, c, o in turns:
        starts.append(len(flat))
        lens.append(len(o))
        for f in o:
            flat.append(pos[f])
        combos.append(c)
        times.append(t)
    extra = {"life": np.array(life)} if all_lives else {}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(dst + ".tmp.npz", frames=r2["frames"][flat], obs_start=np.array(starts),
                        obs_len=np.array(lens), combo=np.array(combos), t=np.array(times), **extra)
    os.replace(dst + ".tmp.npz", dst)
    return dict(play, split=split, n_turns=len(turns), n_frames=len(flat))


def pick(plays, n, rng):
    """n plays drawn without replacement, kept in their original order."""
    out = []
    for i in sorted(rng.choice(len(plays), min(n, len(plays)), replace=False)):
        out.append(plays[i])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="sub-01")
    ap.add_argument("--max-train", type=int, default=300)
    ap.add_argument("--max-val", type=int, default=40)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    from mjg.data import index_plays
    hold = set(json.load(open(f"{CODE}/configs/holdout_sessions.json"))["holdout"].get(a.subject, []))
    plays = []
    for p in index_plays(DATA, [a.subject]):
        if p.ses not in hold:
            plays.append(p)
    sessions = set()
    for p in plays:
        sessions.add(p.ses)
    val_ses = set(sorted(sessions)[2::6])
    rng = np.random.default_rng(0)
    tr, va = [], []
    for p in plays:
        if p.ses in val_ses:
            va.append(p)
        else:
            tr.append(p)
    tr = pick(tr, a.max_train, rng)
    va = pick(va, a.max_val, rng)
    jobs = []
    for s, ps in (("train", tr), ("val", va)):
        for p in ps:
            jobs.append((vars(p), s, a.out, a.every))
    print(f"{a.subject}: {len(plays)} plays outside the fMRI hold-out; using {len(tr)} train, {len(va)} val "
          f"(val sessions {sorted(val_ses)})", flush=True)
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.procs) as pool:
        res = pool.map(work, jobs, chunksize=2)
    with open(f"{a.out}/index.jsonl", "w") as fh:
        for r in res:
            fh.write(json.dumps(r) + "\n")
    for s in ("train", "val"):
        n_plays, n_turns, n_frames = 0, 0, 0
        for r in res:
            if r["split"] == s and r.get("n_turns"):
                n_plays += 1
                n_turns += r["n_turns"]
                n_frames += r["n_frames"]
        print(s, n_plays, "plays,", n_turns, "turns,", n_frames, "frames")


if __name__ == "__main__":
    main()
