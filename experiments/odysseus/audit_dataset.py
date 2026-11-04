"""Audit of the Odysseus SFT dataset (build_dataset.py output): drop rows whose perception is wrong.

The source dataset (--src, default results/odysseus_sft/v1) is never modified. Everything goes to --out
(default results/odysseus_sft/v1_audited), so every decision can be traced and undone:

  auto.jsonl     pass 1 (`--stage auto`): checks against the game's RAM at the turn start (ram.npy of the
                 episode), one line per row: {id, keep, reasons}. Rules:
                   airborne       says Mario is standing while RAM says he is jumping / falling (0x1D != 0)
                   no_enemies     says there are no enemies while an enemy is on screen (RAM enemy slots)
                   ghost_enemy    names an enemy (Goomba, Koopa, ...) while no enemy is on screen
                   wrong_world    names a world-level other than the episode's
                   stuck          4th and later turn of a streak where Mario does not move
  visual.jsonl   pass 2: keep / drop calls from looking at the screens of the rows pass 1 kept (written by a human
                 reviewer, same format, plus `note`)
  audit.jsonl    pass 3 (`--stage final`): every row of the source with its final decision and which pass made it
  train/val/test.jsonl   kept rows, unchanged; `image` points into the source folder's screens
  summary.json   counts per rule, split and level

  python audit_dataset.py --stage auto      # emulator env ($MJG_PY: numpy + mjg)
  python audit_dataset.py --stage final
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
ROOT = os.environ.get("MJG_ROOT", ".")
SRC = f"{ROOT}/results/odysseus_sft/v1"
OUT = f"{ROOT}/results/odysseus_sft/v1_audited"
EPISODES = f"{ROOT}/results/odysseus_play/v1"

# RAM addresses (mario.stimuli data.json)
A = dict(player_x_posHi=0x6D, player_x_posLo=0x86, player_y_pos=0xCE, player_y_screen=0xB5, player_size=0x754,
         xscrollHi=0x71A, xscrollLo=0x71C, jump_airborne=0x1D)
for i in range(6):
    A.update({f"enemy_drawn{15 + i}": 0x0F + i, f"enemy_id_{i}": 0x16 + i, f"enemy_x_{i}": 0x87 + i,
              f"enemy_pageloc_{i}": 0x6E + i, f"enemy_y_{i}": 0xCF + i})
for i in range(5):
    A[f"enemy_y_high_{i}"] = 0xB6 + i

ENEMY_WORDS = r"goomba|koopa|turtle|piranha|bullet bill|hammer|lakitu|spiny|beetle|cheep|blooper|bowser|enem(y|ies)"
NEG = re.compile(r"\b(no|not|none|without|any)\b")
STAND = re.compile(r"\b(standing|stands|is on the ground)\b", re.I)
AIR_WORDS = re.compile(r"\b(jump|jumping|mid-?air|in the air|falling|airborne|leaping)\b", re.I)
NO_ENEMY = re.compile(r"\bno (visible |immediate |other )?(enem(y|ies)|goombas?|koopas?|threats?)\b"
                      r"|without (any )?enem", re.I)


def sentences(text):
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        if s.strip():
            out.append(s)
    return out


def enemies_on_screen(ram):
    from mjg.objects import oracle_objects, TYPE_GOOMBA, TYPE_KOOPA, TYPE_ENEMY
    values = {}
    for k, a in A.items():
        values[k] = int(ram[a])
    out = []
    for o in oracle_objects(values):
        if o.type in (TYPE_GOOMBA, TYPE_KOOPA, TYPE_ENEMY):
            out.append(o)
    return out


def check(row, ram):
    p = row["perception"]
    reasons = []
    if int(ram[A["jump_airborne"]]) != 0 and STAND.search(p) and not AIR_WORDS.search(p):
        reasons.append("airborne")
    enemies = enemies_on_screen(ram)
    if enemies and NO_ENEMY.search(p):
        reasons.append("no_enemies")
    positive = []  # sentences that name an enemy without negating it
    for s in sentences(p):
        if re.search(ENEMY_WORDS, s, re.I) and not NEG.search(s.lower()):
            positive.append(s)
    if not enemies and positive:
        reasons.append("ghost_enemy")
    w = re.search(r"world (\d)\s*-\s*(\d)", p, re.I)
    if w and f"w{w.group(1)}l{w.group(2)}" != row["level"]:
        reasons.append("wrong_world")
    return reasons


def read_jsonl(path):
    rows = []
    for line in open(path):
        rows.append(json.loads(line))
    return rows


def load_rows(src):
    rows = []
    for sp in ("train", "val", "test"):
        rows += read_jsonl(os.path.join(src, f"{sp}.jsonl"))
    return rows


def by_id(path):
    out = {}
    for o in read_jsonl(path):
        out[o["id"]] = o
    return out


def stage_auto(a):
    rows = load_rows(a.src)
    by_ep = defaultdict(list)
    for r in rows:
        by_ep[(r["level"], r["run"])].append(r)
    out = []
    for (lv, run), rs in sorted(by_ep.items()):
        ram = np.load(os.path.join(EPISODES, lv, f"run-{run:03d}", "ram.npy"))
        streak, prev_x = 0, None
        for r in sorted(rs, key=lambda r: r["turn"]):
            reasons = check(r, ram[r["turn"] - 1])
            if r["x_after"] == r["x_before"] and r["x_before"] == prev_x:
                streak += 1
            elif r["x_after"] == r["x_before"]:
                streak = 1
            else:
                streak = 0
            prev_x = r["x_after"]
            if streak >= 4:
                reasons.append("stuck")
            out.append(dict(id=r["id"], keep=not reasons, reasons=reasons))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "auto.jsonl"), "w") as fh:
        for o in out:
            fh.write(json.dumps(o) + "\n")
    c = Counter()
    n_keep = 0
    for o in out:
        n_keep += int(o["keep"])
        for x in o["reasons"]:
            c[x] += 1
    print(f"{len(out)} rows: kept {n_keep}, dropped {len(out) - n_keep}; "
          f"rule hits (a row can hit several): {dict(c.most_common())}")


def decide(r, au, vi, reviewed):
    """Final decision on one row: pass 1 drops first, then the visual decision if there is one."""
    if not au["keep"]:
        return dict(id=r["id"], keep=False, by="auto", reasons=au["reasons"])
    if vi is not None:
        return dict(id=r["id"], keep=vi["keep"], by="visual", level=vi.get("level"), reasons=vi.get("reasons", []),
                    note=vi.get("note"))
    return dict(id=r["id"], keep=True, by="unreviewed" if reviewed else "auto", reasons=[])


def summarize(a, rows, audit, kept):
    n_kept = {}
    per_level = Counter()
    for sp, v in kept.items():
        n_kept[sp] = len(v)
        for r in v:
            per_level[r["level"]] += 1
    dropped_by, reasons, visual_levels = Counter(), Counter(), Counter()
    unreviewed = 0
    for d in audit:
        if not d["keep"]:
            dropped_by[d["by"]] += 1
            for x in d["reasons"]:
                reasons[x] += 1
        if d["by"] == "unreviewed":
            unreviewed += 1
        if d["by"] == "visual":
            visual_levels[d.get("level")] += 1
    return dict(source=a.src, rows=len(rows), kept=n_kept, dropped_by=dict(dropped_by),
                reasons=dict(reasons.most_common()), kept_per_level=dict(per_level), unreviewed=unreviewed,
                visual_levels=dict(visual_levels))


def stage_final(a):
    rows = load_rows(a.src)
    auto = by_id(os.path.join(a.out, "auto.jsonl"))
    vpath = os.path.join(a.out, "visual.jsonl")
    visual = by_id(vpath) if os.path.exists(vpath) else {}
    audit, kept = [], defaultdict(list)
    for r in rows:
        d = decide(r, auto[r["id"]], visual.get(r["id"]), bool(visual))
        audit.append(d)
        if d["keep"]:
            r = dict(r, image=os.path.join(os.path.relpath(a.src, a.out), r["image"]))
            kept[r["split"]].append(r)
    with open(os.path.join(a.out, "audit.jsonl"), "w") as fh:
        for d in audit:
            fh.write(json.dumps(d) + "\n")
    for sp in ("train", "val", "test"):
        with open(os.path.join(a.out, f"{sp}.jsonl"), "w") as fh:
            for r in kept[sp]:
                fh.write(json.dumps(r) + "\n")
    summary = summarize(a, rows, audit, kept)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["auto", "final"], required=True)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.stage == "auto":
        stage_auto(args)
    else:
        stage_final(args)
