"""Full NES RAM (2 KB) for every frame of every human play, by replaying the .bk2 files (CPU only).

The V-JEPA .game.npz files keep only Mario's RAM variables (mjg.emulator.RAM_VARS). This pass keeps the
whole RAM so enemies, objects and level tiles around Mario can be read later without another replay.

  python scripts/data/extract_full_ram.py --shard 0 --nshards 1 [--subjects sub-01 ...] [--plays <play_id> ...]
  python scripts/data/extract_full_ram.py --check <play_id>     # compare with the V-JEPA .game.npz of that play

Output: $MJG_ROOT/features/ram_full/<play_id>.npz (compressed) with
  ram          uint8 [n_frames, 2048]  RAM after the emulator applied the input of frame k (= screen k)
  buttons      bool  [n_frames, n_buttons]  inputs applied, env.buttons order
  button_names       env.buttons
and <play_id>.json (play metadata, frame count, seconds). Resumable: plays with a .json are skipped.
Same replay loop as mjg.emulator.replay_bk2 (frame k = the k-th movie step), without rendering frames.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from mjg import data as D  # noqa: E402
from mjg import emulator as E  # noqa: E402

SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]


def replay_full_ram(path, integration_root):
    retro = E.add_integration(integration_root)
    movie = retro.Movie(path)
    movie.step()
    env = retro.make(game=movie.get_game(), state=retro.State.NONE,
                     inttype=retro.data.Integrations.CUSTOM_ONLY, use_restricted_actions=retro.Actions.ALL,
                     players=movie.players, render_mode=None)
    try:
        env.initial_state = movie.get_state()
        env.reset()
        nb = env.num_buttons
        ram, buttons = [], []
        while movie.step():
            keys = []
            for i in range(nb):
                keys.append(movie.get_key(i, 0))
            env.step(keys)
            buttons.append(keys)
            ram.append(np.array(env.get_ram(), dtype=np.uint8))
        return np.stack(ram), np.asarray(buttons, dtype=bool), list(env.buttons)
    finally:
        env.close()
        del movie


def check(out_dir, features, play_id):
    """Buttons must equal the .game.npz buttons; each Mario RAM variable must equal some RAM column."""
    f = np.load(os.path.join(out_dir, play_id + ".npz"))
    g = np.load(os.path.join(features, play_id + ".game.npz"))
    ram = f["ram"].astype(np.int64)                        # load once (npz members decompress on every access)
    ok = ram.shape[0] == len(g["buttons"]) and bool((f["buttons"] == g["buttons"]).all())
    print(f"{play_id}: frames {ram.shape[0]} vs {len(g['buttons'])}, buttons identical: {ok}")
    for k in g.files:
        if not k.startswith("ram_"):
            continue
        v = g[k].astype(np.int64)
        cols = np.flatnonzero((ram == v[:, None]).all(0)).tolist() if len(v) == len(ram) else []
        addresses = []
        for c in cols[:4]:
            addresses.append(hex(c))
        print(f"  {k:22s} matches RAM address(es) {addresses or 'NONE (multi-byte or derived)'}")
    return ok


def check_all(out_dir, features, play_ids):
    """check() on each play, stopping at the first that fails."""
    for p in play_ids:
        if not check(out_dir, features, p):
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("MJG_DATA"))
    ap.add_argument("--integration", default=os.environ.get("MJG_INTEGRATION"))
    ap.add_argument("--out", default=os.path.join(os.environ.get("MJG_ROOT", "."), "features", "ram_full"))
    ap.add_argument("--features", default=os.path.join(os.environ.get("MJG_ROOT", "."), "features",
                                                       "vjepa21G_c16s8_f8_g8"))
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    ap.add_argument("--plays", nargs="*", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--check", nargs="*", default=None, help="play ids to compare with .game.npz, then exit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.check is not None:
        sys.exit(0 if check_all(a.out, a.features, a.check) else 1)

    plays = D.index_plays(a.data, a.subjects)
    plays.sort(key=lambda p: p.play_id)
    if a.plays:
        wanted = set(a.plays)
        selected = []
        for p in plays:
            if p.play_id in wanted:
                selected.append(p)
        plays = selected
    mine = plays[a.shard::a.nshards]
    todo = []
    for p in mine:
        if not os.path.exists(os.path.join(a.out, p.play_id + ".json")):
            todo.append(p)
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} plays, {len(todo)} to do", flush=True)
    for i, p in enumerate(todo):
        t0 = time.time()
        ram, buttons, names = replay_full_ram(p.bk2, a.integration)
        tmp = os.path.join(a.out, p.play_id + ".tmp.npz")
        name_strs = []
        for b in names:
            name_strs.append(str(b))
        np.savez_compressed(tmp, ram=ram, buttons=buttons, button_names=np.array(name_strs))
        os.replace(tmp, os.path.join(a.out, p.play_id + ".npz"))
        meta = dict(play_id=p.play_id, sub=p.sub, ses=p.ses, level=p.level, rep=p.rep, bk2=p.bk2,
                    n_frames=int(ram.shape[0]), ram_bytes=int(ram.shape[1]), seconds=round(time.time() - t0, 1))
        json.dump(meta, open(os.path.join(a.out, p.play_id + ".json"), "w"), indent=1)
        print(f"[{i + 1}/{len(todo)}] {p.play_id}: {ram.shape[0]} frames in {meta['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
