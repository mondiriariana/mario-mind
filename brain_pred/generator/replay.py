"""Replay a recorded play through the same emulator: every frame (RGB) and the RAM after every frame.

A play folder from glimmer_play.py / odysseus play.py stores the per-frame inputs (inputs.npy) but not the screens,
and RAM only once per turn; replaying the inputs from the level's start state reproduces the play frame by frame.
Check: the RAM at the end of the replay must equal the last RAM the player recorded.

  python replay.py --play <play dir> --out <file.npz>     (env with stable-retro: $MJG_PY)
Output: frames uint8 [n_frames + 1, 224, 240, 3] (CNeuroMod integration screen; frame 0 = after reset),
ram uint8 [n_frames + 1, 2048], lowlevel float32 [n_frames, 3] (luminance, optical flow, audio envelope after each
input: brain_pred/features/lowlevel.py, the same definition as for the human plays), button_names, level.
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "features"))

ROOT = os.environ.get("MJG_ROOT", ".")
CNEUROMOD = os.environ.get("MJG_CNEUROMOD", os.path.join(ROOT, "cneuromod"))
INTEGRATION = os.environ.get("MJG_INTEGRATION", os.path.join(CNEUROMOD, "mario.stimuli"))


def replay(play_dir, integration):
    from mjg import emulator as E
    from lowlevel import LowLevel, env_audio
    meta = json.load(open(f"{play_dir}/meta.json"))
    inputs = np.load(f"{play_dir}/inputs.npy")
    w, l = meta["level"][1], meta["level"][3]
    env = E.make_env(integration, state=f"Level{w}-{l}")
    if meta.get("start_bk2"):  # a scenario (glimmer_play --scenarios): the game started from a human play's start state
        import retro
        movie = retro.Movie(meta["start_bk2"])
        movie.step()
        env.initial_state = movie.get_state()
    obs, _ = env.reset()
    frames, rams, ll, low = [obs], [np.array(env.get_ram(), np.uint8)], LowLevel(), []
    ll.step(obs)  # previous screen for the first frame's optical flow
    for row in inputs:
        a = [0] * env.num_buttons
        pressed = []
        for x in row:
            pressed.append(int(x))
        a[:len(row)] = pressed
        obs, *_ = env.step(a)
        frames.append(obs)
        rams.append(np.array(env.get_ram(), np.uint8))
        low.append(ll.step(obs, env_audio(env)))
    env.close()
    return np.stack(frames), np.stack(rams), meta, np.stack(low)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--integration", default=INTEGRATION)
    a = ap.parse_args()
    frames, rams, meta, low = replay(a.play, a.integration)
    rec = np.load(f"{a.play}/ram.npy")
    ok = bool(np.array_equal(rams[-1], rec[-1]))
    np.savez_compressed(a.out, frames=frames, ram=rams, lowlevel=low, button_names=np.array(meta["button_names"]),
                        level=meta["level"], replay_matches_record=ok)
    print(f"{a.out}: {len(frames)} frames {frames.shape[1:]}; final RAM matches the recorded play: {ok}")


if __name__ == "__main__":
    main()
