"""Index of every eligible moment of the human plays, with labels that describe each situation.

VLA-JEPA's frame set (experiments/vjepa_spark_combo/build_set.py) is drawn from this index.

A moment is a screen k of a human play where Mario is in control:
  player_state (0x0E) == 8, game timer running (not a start / respawn freeze), >= 3 s after the timer of
  the current life started.
Moments are taken on a regular 2 s grid per play (random offset per play, fixed seed), snapped to the
V-JEPA rows (every 8 screens) so each moment has its V-JEPA latent. Nothing is hand-picked.

Labels (all from the full RAM of features/ram_full, see scripts/data/extract_full_ram.py):
  where     level; position in level (start / middle / end thirds of Mario's x, relative to the largest
            x seen on that level; 'bonus' when the area pointer 0x750 differs from the level's main area);
            learning stage (early / late: first / second half of that subject's plays of that level, by
            session then rep); subject; held_out (configs/holdout_sessions.json)
  faces     enemy_ahead (active enemy 0..64 px ahead, within 48 px vertically), enemy_near (|dx| <= 64),
            pit_ahead (no solid tile in the two bottom rows of some column 1..4 tiles ahead),
            wall_ahead (solid tile at body height 1..2 tiles ahead), block_in_reach (brick / ? block /
            coin up to 5 tiles above, -1..+2 tiles around Mario); 'nothing_near' when none of these
  mario     airborne (0x1D), speed (0x700: still < 4 <= walk < 0x19 <= run), power (0x756: small/big/fire)
  human     action in the 0.25 s after k: buttons held >= half of the frames, and a coarse class
            (run_right, walk_right, jump, back_left, stop, other)
The situation labels follow the standard SMB RAM map and are approximate.

  python scripts/data/build_moment_index.py --out $MJG_ROOT/results/moment_index/v1

Writes moments.csv (one row per moment) and coverage.json (counts per label and per cell = level x position x
learning stage, moments outside the held-out sessions only). CPU only (slurm/build_moment_index.sbatch).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

FPS = 60
GRID = 120                      # 2 s between moments of one play
VJEPA_STEP = 8                  # V-JEPA rows every 8 screens
WINDOW = 15                     # 0.25 s human-action window
TILE_BASE, PAGE_SIZE, N_ROWS = 0x500, 0xD0, 13
NON_SOLID = {0x00, 0x24, 0x25, 0x26, 0x5F, 0x60, 0xC2, 0xC3}
REACH_TILES = set(range(0x51, 0x5F)) | {0xC0, 0xC1, 0xC2, 0xC3}
POSITIONS = ("start", "middle", "end")
POWER = {0: "small", 1: "big", 2: "fire"}
FACES_KEYS = ("enemy_ahead", "enemy_near", "pit_ahead", "wall_ahead", "block_in_reach", "nothing_near")


def timer(ram):
    """Game timer from its three decimal digits (0x7F8-0x7FA)."""
    d = ram[:, 0x7F8:0x7FB].astype(int)
    return d[:, 0] * 100 + d[:, 1] * 10 + d[:, 2]


def eligible(ram, min_life_s=3.0):
    """Boolean [n]: Mario in control (see module doc)."""
    n = len(ram)
    t, lives, st = timer(ram), ram[:, 0x75A].astype(int), ram[:, 0x0E]
    ok = np.zeros(n, bool)
    ok[90:] = (st[90:] == 8) & (t[90:] < t[:-90])
    reset = np.zeros(n, bool)
    reset[1:] = (np.diff(t) > 0) | (np.diff(lives) != 0)
    life_start = np.maximum.accumulate(np.where(reset | (np.arange(n) == 0), np.arange(n), 0))
    ticking = np.full(n, -1)
    for s in np.unique(life_start):                       # first screen of each life with the timer running
        seg = np.arange(s, n)[life_start[s:] == s]
        run = seg[t[seg] < t[s]]
        ticking[seg] = run[0] if len(run) else n
    return ok & (np.arange(n) - ticking >= int(min_life_s * FPS)), life_start


def mario(ram, k):
    x = int(ram[k, 0x6D]) * 256 + int(ram[k, 0x86])
    return x, int(ram[k, 0xCE])


def tile(ram, k, x, y):
    """Metatile at level pixel (x, y) in the 2-page buffer, or None when off the buffer rows."""
    row = (y - 32) // 16
    if not 0 <= row < N_ROWS:
        return None
    page, col = (x // 256) % 2, (x % 256) // 16
    return int(ram[k, TILE_BASE + page * PAGE_SIZE + row * 16 + col])


def pit_in_column(ram, k, bx):
    """True when both bottom rows of the tile column at level pixel bx are empty."""
    for r in (N_ROWS - 2, N_ROWS - 1):
        b = tile(ram, k, bx, 32 + 16 * r)
        if b is None or b not in NON_SOLID:
            return False
    return True


def faces(ram, k):
    mx, my = mario(ram, k)
    lab = dict(enemy_ahead=False, enemy_near=False, pit_ahead=False, wall_ahead=False, block_in_reach=False)
    for i in range(5):                                    # enemy slots
        if ram[k, 0x0F + i] == 0 or ram[k, 0xB6 + i] != 1:
            continue
        ex, ey = int(ram[k, 0x6E + i]) * 256 + int(ram[k, 0x87 + i]), int(ram[k, 0xCF + i])
        dx, dy = ex - mx, ey - my
        lab["enemy_near"] |= abs(dx) <= 64
        lab["enemy_ahead"] |= 0 < dx <= 64 and abs(dy) <= 48
    for c in range(1, 5):                                 # pit: both bottom rows empty in a column ahead
        if pit_in_column(ram, k, mx + 8 + 16 * c):
            lab["pit_ahead"] = True
    for c in (1, 2):                                      # wall: solid at body height
        for yy in (my + 8, my - 8):
            tl = tile(ram, k, mx + 8 + 16 * c, yy)
            lab["wall_ahead"] |= tl is not None and tl not in NON_SOLID
    for c in (-1, 0, 1, 2):
        for r in range(1, 6):
            tl = tile(ram, k, mx + 8 + 16 * c, my - 16 * r)
            lab["block_in_reach"] |= tl is not None and tl in REACH_TILES
    anything = False
    for v in lab.values():
        if v:
            anything = True
            break
    lab["nothing_near"] = not anything
    return lab


def human(buttons, names, k):
    w = buttons[k + 1: k + 1 + WINDOW]
    frac = dict(zip(names, w.mean(0))) if len(w) else {}
    held = []
    for n, f in frac.items():
        if f >= 0.5:
            held.append(n)
    h = set(held)
    if "A" in h:
        cls = "jump"
    elif "RIGHT" in h:
        cls = "run_right" if "B" in h else "walk_right"
    elif "LEFT" in h:
        cls = "back_left"
    elif not h or h <= {"B"}:
        cls = "stop"
    else:
        cls = "other"
    return held, cls


def read_holdout(path):
    """{(subject, session)} of the sessions held out for the fMRI analysis."""
    with open(path) as fh:
        holdout = json.load(fh)["holdout"]
    held = set()
    for sub, sessions in holdout.items():
        for ses in sessions:
            held.add((sub, ses))
    return held


def read_metas(ram_dir):
    """Play metadata of every ram_full/<play>.json, sorted by play id."""
    metas = []
    for p in glob.glob(os.path.join(ram_dir, "*.json")):
        with open(p) as fh:
            metas.append(json.load(fh))
    return sorted(metas, key=lambda m: m["play_id"])


def learning_stages(metas):
    """play_id -> 'early' / 'late': first / second half of the subject's plays of that level (by session, rep)."""
    by_sl = defaultdict(list)
    for m in metas:
        by_sl[(m["sub"], m["level"])].append(m)
    stage = {}
    for g in by_sl.values():
        g.sort(key=lambda m: (m["ses"], m["rep"]))
        for i, m in enumerate(g):
            stage[m["play_id"]] = "early" if i < len(g) / 2 else "late"
    return stage


def moment_row(m, ram, buttons, names, k, life_start, held, stage):
    """One moments.csv row (without its position label, which needs every play of the level)."""
    x, _ = mario(ram, k)
    hb, hc = human(buttons, names, k)
    sp = int(ram[k, 0x700])
    row = dict(
        play_id=m["play_id"], sub=m["sub"], ses=m["ses"], level=m["level"], rep=m["rep"],
        held_out=(m["sub"], m["ses"]) in held, stage=stage[m["play_id"]], screen=k,
        vjepa_row=k // VJEPA_STEP, t_s=round(k / FPS, 2), life_start=int(life_start[k]), x=x,
        area=int(ram[k, 0x750]), airborne=bool(ram[k, 0x1D]),
        speed="still" if sp < 4 else "walk" if sp < 0x19 else "run",
        power=POWER.get(int(ram[k, 0x756]), str(int(ram[k, 0x756]))))
    for kk, v in faces(ram, k).items():
        row[kk] = bool(v)
    row["human_action"] = "+".join(hb) or "none"
    row["human_class"] = hc
    return row


def count_by(rows, key):
    c = Counter()
    for r in rows:
        c[r[key]] += 1
    return c


def total_of(rows, key):
    n = 0
    for r in rows:
        n += r[key]
    return int(n)


def coverage(rows, metas, settings, main_area, level_max):
    """Counts per label and per cell (level x position x learning stage), held-out sessions excluded."""
    el = []
    for r in rows:
        if not r["held_out"]:
            el.append(r)
    cells = Counter()
    for r in el:
        cells[(r["level"], r["position"], r["stage"])] += 1
    below_30 = []
    for c, n in cells.items():
        if n < 30:
            below_30.append("|".join(c))
    face_counts = {}
    for k in FACES_KEYS:
        face_counts[k] = total_of(el, k)
    cell_counts = {}
    for c, n in sorted(cells.items()):
        cell_counts["|".join(c)] = n
    return dict(
        settings=settings, n_plays=len(metas), n_moments=len(rows), n_eligible_not_held_out=len(el),
        n_held_out=len(rows) - len(el), n_cells=len(cells),
        cells_min=min(cells.values()) if cells else 0, cells_below_30=sorted(below_30),
        by_subject=dict(count_by(el, "sub")), by_level=dict(sorted(count_by(el, "level").items())),
        by_position=dict(count_by(el, "position")), by_stage=dict(count_by(el, "stage")),
        faces=face_counts, airborne=total_of(el, "airborne"), by_speed=dict(count_by(el, "speed")),
        by_power=dict(count_by(el, "power")), by_human_class=dict(count_by(el, "human_class")),
        main_area=main_area, level_max_x=dict(level_max), cells=cell_counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.environ.get("MJG_ROOT", ".")
    ap.add_argument("--ram", default=f"{root}/features/ram_full")
    ap.add_argument("--holdout", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                      "configs", "holdout_sessions.json"))
    ap.add_argument("--out", default=f"{root}/results/moment_index/v1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    held = read_holdout(a.holdout)
    metas = read_metas(a.ram)
    stage = learning_stages(metas)

    rows, level_max, area_votes = [], defaultdict(int), defaultdict(Counter)
    for m in metas:
        # allow_pickle: these are this pipeline's own files, and button_names may hold None
        z = np.load(os.path.join(a.ram, m["play_id"] + ".npz"), allow_pickle=True)
        ram, buttons = z["ram"], z["buttons"]
        names = []
        for b in z["button_names"]:
            names.append(str(b))
        ok, life_start = eligible(ram)
        off = int(rng.integers(0, GRID // VJEPA_STEP)) * VJEPA_STEP
        for k in range(off, len(ram) - WINDOW - 1, GRID):
            if not ok[k]:
                continue
            row = moment_row(m, ram, buttons, names, k, life_start, held, stage)
            area_votes[m["level"]][row["area"]] += 1
            level_max[m["level"]] = max(level_max[m["level"]], row["x"])
            rows.append(row)
    main_area = {}
    for lv, c in area_votes.items():
        main_area[lv] = c.most_common(1)[0][0]
    for r in rows:
        if r["area"] != main_area[r["level"]]:
            r["position"] = "bonus"
        else:
            r["position"] = POSITIONS[min(int(3 * r["x"] / max(level_max[r["level"]], 1)), 2)]

    os.makedirs(a.out, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with open(os.path.join(a.out, "moments.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    cov = coverage(rows, metas, vars(a), main_area, level_max)
    with open(os.path.join(a.out, "coverage.json"), "w") as fh:
        json.dump(cov, fh, indent=1)
    n_el = cov["n_eligible_not_held_out"]
    print(f"{len(rows)} moments from {len(metas)} plays ({n_el} usable, {len(rows) - n_el} held out), "
          f"{cov['n_cells']} cells, smallest cell {cov['cells_min']} -> {a.out}")


if __name__ == "__main__":
    main()
