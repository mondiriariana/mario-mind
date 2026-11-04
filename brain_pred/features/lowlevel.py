"""Low-level confounds per frame: screen luminance, optical flow, audio envelope (CPU).

The same three signals CNeuroMod ships as gamelogs/*_lowlevel.npy (mario dataset, dev_replays branch; used as GLM
confounds in courtois-neuromod/mario.tutorials), computed here from the replays, the same way for a person's play
(.bk2) and an agent's play (generator/replay.py):
  luminance       mean grey level of the screen, 0-1
  optical_flow    mean magnitude of the dense optical flow (Farneback) from the previous screen, pixels per frame,
                  on the screen at half resolution (112 x 120 grey)
  audio_envelope  RMS of the emulator's audio samples of the frame (int16 scale / 32768)
They go into the encoders' nuisance band (encode.py --lowlevel), so a feature band only gets credit for what these
do not explain (brightness changes, scrolling, sound).

  python lowlevel.py --subjects sub-03 --procs 32     (env with stable-retro + opencv: $MJG_PY)
Output: $MJG_ROOT/features/lowlevel/<play_id>.npy float32 [n_frames, 3], indexed like features/ram_full (frame k =
after the input of frame k). Resumable per play.
"""
import argparse, os, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

ROOT = os.environ.get("MJG_ROOT", ".")
CNEUROMOD = os.environ.get("MJG_CNEUROMOD", os.path.join(ROOT, "cneuromod"))
OUT = f"{ROOT}/features/lowlevel"
NAMES = ("luminance", "optical_flow", "audio_envelope")  # the output columns, in order


class LowLevel:
    """Feed screens (and the frame's audio) one at a time; step() returns the 3 values of that frame."""

    def __init__(self):
        import cv2
        self.cv2, self.prev = cv2, None

    def step(self, screen, audio=None):
        g = self.cv2.cvtColor(np.asarray(screen, np.uint8), self.cv2.COLOR_RGB2GRAY)
        small = self.cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=self.cv2.INTER_AREA)
        flow = 0.0
        if self.prev is not None:
            f = self.cv2.calcOpticalFlowFarneback(self.prev, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = float(np.sqrt((f ** 2).sum(-1)).mean()) * 2  # back to full-resolution pixels
        self.prev = small
        a = np.asarray(audio, np.float32) / 32768 if audio is not None and np.size(audio) else np.zeros(1, np.float32)
        return np.array([g.mean() / 255, flow, np.sqrt((a ** 2).mean())], np.float32)


def env_audio(env):
    """The audio samples the emulator produced for the last frame (stable-retro), or None."""
    try:
        return env.em.get_audio()
    except Exception:
        return None


def replay_bk2_lowlevel(path, integration_root):
    """Same replay loop as scripts/data/extract_full_ram.py (frame k = the k-th movie step) -> [n_frames, 3]."""
    from mjg import emulator as E
    retro = E.add_integration(integration_root)
    movie = retro.Movie(path)
    movie.step()
    env = retro.make(game=movie.get_game(), state=retro.State.NONE,
                     inttype=retro.data.Integrations.CUSTOM_ONLY, use_restricted_actions=retro.Actions.ALL,
                     players=movie.players, render_mode=None)
    try:
        env.initial_state = movie.get_state()
        env.reset()
        nb, ll, out = env.num_buttons, LowLevel(), []
        while movie.step():
            keys = []
            for i in range(nb):
                keys.append(movie.get_key(i, 0))
            obs, *_ = env.step(keys)
            out.append(ll.step(obs, env_audio(env)))
        return np.stack(out)
    finally:
        env.close()
        del movie


def one_play(job):
    play_id, bk2, integration = job
    dst = f"{OUT}/{play_id}.npy"
    if os.path.exists(dst):
        return play_id, "exists"
    t0 = time.time()
    x = replay_bk2_lowlevel(bk2, integration)
    np.save(dst + ".tmp.npy", x)
    os.replace(dst + ".tmp.npy", dst)
    return play_id, f"{len(x)} frames, audio {'yes' if x[:, 2].any() else 'NONE'} ({time.time() - t0:.0f}s)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", default=["sub-03"])
    ap.add_argument("--plays", nargs="*", default=None, help="only these play ids")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--data", default=os.environ.get("MJG_DATA", f"{CNEUROMOD}/data/mario"))
    ap.add_argument("--integration", default=os.environ.get("MJG_INTEGRATION", f"{CNEUROMOD}/mario.stimuli"))
    a = ap.parse_args()
    from mjg.data import index_plays
    os.makedirs(OUT, exist_ok=True)
    plays = index_plays(a.data, a.subjects)
    if a.plays:
        wanted = set(a.plays)
        chosen = []
        for p in plays:
            if p.play_id in wanted:
                chosen.append(p)
        plays = chosen
    jobs = []
    for p in plays:
        jobs.append((p.play_id, p.bk2, a.integration))
    t0 = time.time()
    with Pool(a.procs) as pool:
        for i, (pid, msg) in enumerate(pool.imap_unordered(one_play, jobs)):
            print(f"[{i + 1}/{len(jobs)}] {pid}: {msg} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
