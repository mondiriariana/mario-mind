"""Turn the Odysseus self-play episodes (play.py output) into the reasoning-trace dataset (audited by
audit_dataset.py, then used by experiments/glimmer_imitation/train_sft.py --traces).

Rules:
  - format filter: reply parsed (an <answer> list of valid buttons), non-empty <perception> and
    <reasoning>, not cut at the token limit, not looping. Noise is otherwise accepted.
  - de-duplication of near-identical level-start traces: turns that saw the same game state (identical
    RAM at the turn start, e.g. the first turn of every run of a level) and whose text is near-identical
    (similarity >= --dup-sim) are kept once, within each split.
  - split by whole episode, stratified by level: per level the runs are shuffled (fixed seed) into
    6 train / 1 val / 1 test.
  - no per-step matching to the human.

Episodes do not store screens, but inputs.npy replays them exactly: the full run replays every kept
episode and saves the screen each turn saw (the native 224x240 image; Odysseus saw it upscaled to 1280x1152,
nearest), checking the emulator's RAM against ram.npy at every turn start.

  python build_dataset.py --check                 # anywhere (no emulator): completeness, filters, counts
  python build_dataset.py                         # emulator env ($MJG_PY): + screens + final files
  sbatch experiments/odysseus/build_dataset.sbatch

Output in --out: {train,val,test}.jsonl (one line per kept turn), screens/<level>/run-NNN/turn-NNN.png,
summary.json (counts per level and split, every dropped turn with its reason). Refuses to finalise while
episodes are missing or inconsistent, unless --allow-partial.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
ROOT = os.environ.get("MJG_ROOT", ".")
CNEUROMOD = os.environ.get("MJG_CNEUROMOD", os.path.join(ROOT, "cneuromod"))
SRC = f"{ROOT}/results/odysseus_play/v1"
OUT = f"{ROOT}/results/odysseus_sft/v1"
LEVELS = ["w1l1", "w1l2", "w1l3", "w2l1", "w2l3", "w3l1", "w3l2", "w3l3", "w4l1", "w4l2", "w4l3",
          "w5l1", "w5l2", "w5l3", "w6l1", "w6l2", "w6l3", "w7l1", "w7l3", "w8l1", "w8l2", "w8l3"]
VALID = {"a", "b", "up", "down", "left", "right", "noop"}


# ------------------------------------------------------------------------------------------ loading
def load_episode(d):
    """(meta, steps, ram) or raises ValueError if the episode is incomplete or inconsistent."""
    for f in ("meta.json", "steps.jsonl", "inputs.npy", "ram.npy"):
        if not os.path.exists(os.path.join(d, f)):
            raise ValueError(f"missing {f}")
    meta = json.load(open(os.path.join(d, "meta.json")))
    steps = []
    for i, line in enumerate(open(os.path.join(d, "steps.jsonl"))):
        try:
            steps.append(json.loads(line))
        except json.JSONDecodeError:
            raise ValueError(f"steps.jsonl line {i + 1} is not JSON")
    turns, held = [], 0
    for s in steps:
        turns.append(s["turn"])
        held += s["frames_held"]
    if len(steps) != meta["turns"] or turns != list(range(1, len(steps) + 1)):
        raise ValueError(f"{len(steps)} lines vs {meta['turns']} turns in meta.json, or turns not 1..n")
    ram = np.load(os.path.join(d, "ram.npy"))
    inputs = np.load(os.path.join(d, "inputs.npy"))
    if len(ram) != len(steps) + 1:
        raise ValueError(f"ram.npy has {len(ram)} rows for {len(steps)} turns")
    if len(inputs) != meta["frames"] or meta["start_frames"] + held != len(inputs):
        raise ValueError("inputs.npy length does not match start_frames + frames held")
    return meta, steps, ram


# ------------------------------------------------------------------------------------------ filters
def looping(text, max_line_repeats=3, ngram=8, max_ngram_repeats=4):
    lines = []
    for line in text.splitlines():
        if len(line.strip()) > 20:
            lines.append(line.strip())
    if lines and max(Counter(lines).values()) > max_line_repeats:
        return True
    w = text.split()
    grams = Counter()
    for i in range(len(w) - ngram + 1):
        grams[tuple(w[i:i + ngram])] += 1
    return bool(grams) and max(grams.values()) > max_ngram_repeats


def drop_reason(s, max_new):
    if s.get("error"):
        return f"parse: {s['error']}"
    if s.get("invalid_buttons"):
        return "invalid buttons"
    if not (s.get("perception") or "").strip():
        return "empty perception"
    if not (s.get("reasoning") or "").strip():
        return "empty reasoning"
    if s.get("new_tokens", 0) >= max_new:
        return "cut at the token limit"
    if looping((s.get("perception") or "") + "\n" + (s.get("reasoning") or "")):
        return "looping"
    return None


def split_runs(runs_by_level, seed, n_val=1, n_test=1):
    out = {}
    for lv, runs in sorted(runs_by_level.items()):
        r = sorted(runs)
        random.Random(f"{seed}-{lv}").shuffle(r)
        for i, run in enumerate(r):
            out[(lv, run)] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def dedup(rows, sim):
    """Within each split: same game state (RAM hash) + near-identical text -> keep the first."""
    kept, dropped, seen = [], [], defaultdict(list)
    for r in rows:
        key = (r["split"], r["ram_hash"])
        text = r["perception"] + "\n" + r["reasoning"]
        duplicate = False
        for t in seen[key]:
            if difflib.SequenceMatcher(None, text, t).ratio() >= sim:
                duplicate = True
                break
        if duplicate:
            dropped.append(r)
            continue
        seen[key].append(text)
        kept.append(r)
    return kept, dropped


# ------------------------------------------------------------------------------------------ screens
def replay_screens(job):
    """Replay one episode's inputs; save the screen at every turn start; check RAM against ram.npy."""
    src, dst, level, want_turns, integration = job
    from PIL import Image
    from mjg import emulator as E
    meta, steps, ram = load_episode(src)
    inputs = np.load(os.path.join(src, "inputs.npy"))
    env = E.make_env(integration, state=f"Level{level[1]}-{level[3]}")
    obs, _ = env.reset()
    held = [meta["start_frames"]]
    for s in steps:
        held.append(s["frames_held"])
    starts = np.cumsum(held)[:-1]  # turn t starts at starts[t-1]
    os.makedirs(dst, exist_ok=True)
    k, saved = 0, {}
    try:
        for t, f0 in enumerate(starts, 1):
            while k < f0:
                pressed = []
                for x in inputs[k]:
                    pressed.append(int(x))
                a = [0] * env.num_buttons
                a[: inputs.shape[1]] = pressed
                obs, *_ = env.step(a)
                k += 1
            if not np.array_equal(np.array(env.get_ram(), dtype=np.uint8), ram[t - 1]):
                return src, None, f"RAM mismatch at turn {t}"
            if t in want_turns:
                p = os.path.join(dst, f"turn-{t:03d}.png")
                Image.fromarray(obs).save(p)
                saved[t] = p
    finally:
        env.close()
    return src, saved, None


# ------------------------------------------------------------------------------------------ main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--levels", nargs="+", default=LEVELS)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dup-sim", type=float, default=0.9)
    ap.add_argument("--integration", default=os.environ.get("MJG_INTEGRATION", f"{CNEUROMOD}/mario.stimuli"))
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--check", action="store_true", help="no emulator, no files: completeness, filters, counts")
    ap.add_argument("--allow-partial", action="store_true", help="finalise even if episodes are missing / broken")
    return ap.parse_args()


def load_episodes(a):
    """-> ({(level, run): (meta, steps, ram)}, {"<level>/run-NNN": problem})."""
    problems, episodes = {}, {}
    for lv in a.levels:
        for r in range(a.runs):
            d = os.path.join(a.src, lv, f"run-{r:03d}")
            try:
                episodes[(lv, r)] = load_episode(d)
            except (ValueError, OSError) as e:
                problems[f"{lv}/run-{r:03d}"] = str(e)
    return episodes, problems


def filter_turns(a, episodes, splits):
    """-> (kept rows, dropped turns with their reason)."""
    rows, drops = [], []
    for (lv, r), (meta, steps, ram) in sorted(episodes.items()):
        for s in steps:
            base = dict(id=f"{lv}/run-{r:03d}/turn-{s['turn']:03d}", level=lv, run=r, turn=s["turn"],
                        split=splits[(lv, r)])
            why = drop_reason(s, meta["max_new_tokens"])
            if why:
                drops.append(dict(base, reason=why))
                continue
            rows.append(dict(base, ram_hash=hashlib.sha1(ram[s["turn"] - 1].tobytes()).hexdigest(),
                             prompt=meta["prompt"], perception=s["perception"].strip(),
                             reasoning=s["reasoning"].strip(), answer=s["answer"].strip(), buttons=s["buttons"],
                             reply=s["reply"], new_tokens=s["new_tokens"], x_before=s["x_before"],
                             x_after=s["x_after"], episode_end=meta["end"]))
    rows, dups = dedup(rows, a.dup_sim)
    for r in dups:
        drops.append(dict(id=r["id"], level=r["level"], run=r["run"], turn=r["turn"], split=r["split"],
                          reason="near-duplicate of a kept trace (same game state)"))
    return rows, drops


def report(a, episodes, rows, drops):
    """Print the counts; -> kept rows per level and split."""
    n_turns = 0
    for e in episodes.values():
        n_turns += len(e[1])
    reasons = Counter()
    for d in drops:
        reasons[d["reason"]] += 1
    print(f"turns: {n_turns}; kept {len(rows)}; dropped {len(drops)}: {dict(reasons.most_common())}")
    by = Counter()
    for r in rows:
        by[r["split"]] += 1
    print("kept per split:", dict(by))
    per_level, totals = {}, {}
    for lv in a.levels:
        c = Counter()
        for r in rows:
            if r["level"] == lv:
                c[r["split"]] += 1
        per_level[lv] = dict(c)
        totals[lv] = sum(per_level[lv].values())
    print("kept per level:", totals)
    return n_turns, per_level


def write_screens(a, rows):
    """Replay the episodes of the kept rows and save their screens -> ({(episode, turn): png}, {episode: problem})."""
    want = defaultdict(set)
    for r in rows:
        want[(r["level"], r["run"])].add(r["turn"])
    jobs = []
    for (lv, run), turns in sorted(want.items()):
        jobs.append((os.path.join(a.src, lv, f"run-{run:03d}"), os.path.join(a.out, "screens", lv, f"run-{run:03d}"),
                     lv, turns, a.integration))
    with Pool(a.procs) as pool:
        res = pool.map(replay_screens, jobs, chunksize=1)
    images, bad = {}, {}
    for src, saved, err in res:
        if err:
            bad[os.path.relpath(src, a.src)] = err
        else:
            for t, p in saved.items():
                images[(os.path.relpath(src, a.src), t)] = p
    return images, bad


def main():
    a = parse_args()
    episodes, problems = load_episodes(a)
    print(f"episodes: {len(episodes)} usable of {len(a.levels) * a.runs}; problems: {len(problems)}")
    for k, v in sorted(problems.items()):
        print(f"  {k}: {v}")

    runs_by_level = {}
    for lv in a.levels:
        runs_by_level[lv] = []
        for (l2, r) in episodes:
            if l2 == lv:
                runs_by_level[lv].append(r)
    splits = split_runs(runs_by_level, a.seed)
    rows, drops = filter_turns(a, episodes, splits)
    n_turns, per_level = report(a, episodes, rows, drops)
    if a.check:
        return
    if problems and not a.allow_partial:
        raise SystemExit("not finalising: episodes missing or inconsistent (or pass --allow-partial)")

    images, bad = write_screens(a, rows)
    if bad:
        print("replay problems (episodes dropped):", bad)
    final = defaultdict(list)
    for r in rows:
        ep = f"{r['level']}/run-{r['run']:03d}"
        if ep in bad:
            continue
        r = dict(r)
        del r["ram_hash"]
        r["image"] = os.path.relpath(images[(ep, r["turn"])], a.out)
        final[r["split"]].append(r)
    os.makedirs(a.out, exist_ok=True)
    for sp in ("train", "val", "test"):
        with open(os.path.join(a.out, f"{sp}.jsonl"), "w") as fh:
            for r in final[sp]:
                fh.write(json.dumps(r) + "\n")
    n_kept, split_of_run = {}, {}
    for sp, v in final.items():
        n_kept[sp] = len(v)
    for (lv, r), sp in sorted(splits.items()):
        split_of_run[f"{lv}/run-{r:03d}"] = sp
    summary = dict(source=a.src, seed=a.seed, dup_sim=a.dup_sim, episodes=len(episodes), episode_problems=problems,
                   replay_problems=bad, turns=n_turns, kept=n_kept, kept_per_level=per_level,
                   split_of_run=split_of_run,
                   image_note="native 224x240 screen; Odysseus saw it upscaled to 1280x1152 (nearest)",
                   dropped=drops)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("written to", a.out, n_kept)


if __name__ == "__main__":
    main()
