"""Super Mario Bros emulation through stable-retro with CNeuroMod's integration.

The integration folder (``SuperMarioBros-Nes/`` holding rom.nes, data.json, the level
states) comes from github.com/courtois-neuromod/mario.stimuli. Its parent directory is
passed as ``integration_root``. The ROM itself is never stored in this repository.

Frame convention used everywhere in this project: ``screen[k]`` is the image returned
after the emulator has applied the controller input of frame ``k``. A ``.bk2`` replay
therefore yields one screen per input line.
"""
from __future__ import annotations

import os
import zipfile

import numpy as np

GAME = "SuperMarioBros-Nes"

# stable-retro returns the NES screen with overscan removed: 224 x 240 x 3, identical in
# geometry to the CNeuroMod mp4 recordings (video frame k == replay screen k, no crop, no
# offset; docs/DATA.md).
SCREEN_SHAPE = (224, 240, 3)

# RAM variables we keep per frame (subset of data.json; see docs/DATA.md).
RAM_VARS = [
    "player_x_posHi", "player_x_posLo", "player_y_pos", "player_y_screen",
    "player_x_speed", "player_y_speed", "player_state", "player_status",
    "lives", "score", "coins", "time", "xscrollHi", "xscrollLo",
    "jump_airborne", "oper_mode", "stage", "world", "powerstate",
]


def _retro():
    import retro  # stable-retro installs the module as ``retro``
    return retro


def add_integration(integration_root: str):
    retro = _retro()
    retro.data.Integrations.add_custom_path(os.path.abspath(integration_root))
    return retro


def make_env(integration_root: str, state: str | None = None):
    """Create an emulator with every NES button available (12 = 9 used + padding).

    ``state`` is a level state name such as ``"Level1-1"``; ``None`` leaves the
    emulator without a state so a caller can set ``env.initial_state`` (bk2 start).
    """
    retro = add_integration(integration_root)
    kw = dict(
        game=GAME,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL,
        render_mode=None,
    )
    kw["state"] = state if state is not None else retro.State.NONE
    return retro.make(**kw)


def full_x(info: dict) -> int:
    """Absolute horizontal position of Mario in the level, in pixels."""
    return int(info["player_x_posHi"]) * 256 + int(info["player_x_posLo"])


def read_bk2_inputs(path: str) -> tuple[list[str], np.ndarray]:
    """Parse the per-frame controller log of a stable-retro ``.bk2`` without a ROM.

    Returns the button names (header order) and a bool array [n_frames, n_buttons].
    """
    with zipfile.ZipFile(path) as z:
        text = z.read("Input Log.txt").decode()
    lines = []
    header = None
    for l in text.splitlines():
        if l.startswith("|"):
            lines.append(l)
        if header is None and l.startswith("P1 "):
            header = l
    names = []
    for n in header.strip("|").split("|"):
        if n:
            names.append(n.replace("P1 ", "").strip())
    rows = []
    for l in lines:
        fields = l.strip("|").split("|")
        keys = fields[-1]
        row = []
        for c in keys:
            row.append(c != ".")
        rows.append(row)
    return names, np.asarray(rows, dtype=bool)


def replay_bk2(path: str, integration_root: str, keep_frame=lambda k: True, ram_vars=None):
    """Replay a human ``.bk2`` through the emulator.

    ``ram_vars`` lists the RAM variables recorded per frame (default ``RAM_VARS``).

    Returns a dict with:
      ``frames``: uint8 [n_kept, H, W, 3] screens for frames where ``keep_frame(k)``
      ``frame_idx``: indices k of the kept frames
      ``buttons``: bool [n_frames, n_buttons] (env.buttons order) inputs actually applied
      ``button_names``: env.buttons
      ``ram``: dict var -> int array [n_frames]
    """
    retro = add_integration(integration_root)
    movie = retro.Movie(path)
    movie.step()
    env = retro.make(
        game=movie.get_game(), state=retro.State.NONE,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL, players=movie.players,
        render_mode=None,
    )
    try:
        env.initial_state = movie.get_state()
        env.reset()
        nb = env.num_buttons
        frames, idx, buttons = [], [], []
        ram_vars = RAM_VARS if ram_vars is None else ram_vars
        ram = {}
        for v in ram_vars:
            ram[v] = []
        k = 0
        while movie.step():
            keys = []
            for i in range(nb):
                keys.append(movie.get_key(i, 0))
            obs, _, _, _, info = env.step(keys)
            buttons.append(keys)
            for v in ram_vars:
                ram[v].append(info.get(v, 0))
            if keep_frame(k):
                frames.append(obs)
                idx.append(k)
            k += 1
        ram_arrays = {}
        for v, a in ram.items():
            ram_arrays[v] = np.asarray(a)
        out = dict(
            frames=np.stack(frames) if frames else np.zeros((0,), np.uint8),
            frame_idx=np.asarray(idx),
            buttons=np.asarray(buttons, dtype=bool),
            button_names=list(env.buttons),
            ram=ram_arrays,
        )
    finally:
        env.close()
        del movie
    return out
