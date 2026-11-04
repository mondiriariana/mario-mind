"""BOLD generator, end to end: a recorded Super Mario Bros. play -> predicted fMRI for one or more subjects.

  play folder (glimmer_play.py / odysseus play.py: inputs.npy, meta.json; hidden.npy with --save-hidden)
    -> replay.py: every frame + the RAM after every frame (the player stores RAM once per turn only)
    -> brain_pred/features: per-TR features (same definitions as the human training runs), including the agent's
       own internal state per TR when the play has hidden.npy: with an agent_* encoder the BOLD is generated from
       the agent's mind, not from the game
    -> brain_pred/prediction: a saved encoder per subject -> predicted BOLD [n_TR, 1000]
Writes <out>.npz: bold__<subject> per subject, the TR length, the encoder used and its held-out mean r.

  python generate_bold.py --play <dir> --subjects sub-01 sub-03 --model gamestate --out <file.npz>
  (env with stable-retro for the replay: $MJG_PY; encoders from encode.py --save $MJG_ROOT/results/bold_generator)
"""
import argparse, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "features"))
sys.path.insert(0, os.path.join(HERE, "..", "prediction"))
sys.path.insert(0, HERE)
from play_features import tr_features, load_play_meta, turn_starts, agent_tr_features, TR  # noqa: E402
from predict import EncoderModel  # noqa: E402

ROOT = os.environ.get("MJG_ROOT", ".")
CNEUROMOD = os.environ.get("MJG_CNEUROMOD", os.path.join(ROOT, "cneuromod"))
INTEGRATION = os.environ.get("MJG_INTEGRATION", os.path.join(CNEUROMOD, "mario.stimuli"))
ENCODERS = f"{ROOT}/results/bold_generator"


def generate(play_dir, subjects, model="gamestate", replay_npz=None, vjepa=None, integration=None):
    """-> {subject: bold [n_TR, 1000]}, {subject: EncoderModel}. replay_npz: reuse a saved replay (frames + ram)."""
    inputs, names, level = load_play_meta(play_dir)
    rp = np.load(replay_npz) if replay_npz and os.path.exists(replay_npz) else None
    if rp is not None and "lowlevel" in rp:
        ram, low = rp["ram"], rp["lowlevel"]
    else:
        from replay import replay
        _, ram, _, low = replay(play_dir, integration or INTEGRATION)
    f = tr_features(inputs, ram[1:], names, level, lowlevel=low)  # ram[1:]: the RAM after each input
    if vjepa is not None:
        f["vjepa"] = vjepa
    if os.path.exists(f"{play_dir}/hidden.npy"):  # the agent's own internal state per turn (glimmer_play --save-hidden)
        f["agent"] = agent_tr_features(np.load(f"{play_dir}/hidden.npy"), turn_starts(play_dir), len(f["buttons"]))
    out, enc = {}, {}
    for sub in subjects:
        enc[sub] = EncoderModel(f"{ENCODERS}/{sub}/{model}.npz")
        if enc[sub].needs_vjepa and "vjepa" not in f:
            raise ValueError(f"encoder {model} needs per-TR V-JEPA features for this play (--vjepa)")
        if enc[sub].needs_agent and "agent" not in f:
            raise ValueError(f"encoder {model} needs the agent's internal state: record the play with --save-hidden")
        out[sub] = enc[sub].predict(f).astype(np.float32)
    return out, enc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play", required=True)
    ap.add_argument("--subjects", nargs="+", default=["sub-03"])
    ap.add_argument("--model", default="gamestate",
                    help="encoder: nuis / gamestate / vjepa4 (game features) or agent_<tag>_d<0-3> (the agent's "
                         "own state)")
    ap.add_argument("--replay", default=None, help="saved replay.py output to reuse")
    ap.add_argument("--vjepa", default=None, help="per-TR V-JEPA 2.1 G features of this play [n_TR, 4, 1664] (.npy)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bold, enc = generate(a.play, a.subjects, a.model, a.replay, np.load(a.vjepa) if a.vjepa else None)
    arrays = {}
    for s, y in bold.items():
        arrays[f"bold__{s}"] = y
    for s, e in enc.items():
        arrays[f"heldout_mean_r__{s}"] = np.float32(np.nanmean(e.r_heldout))
    np.savez_compressed(a.out, tr=TR, model=a.model, **arrays)
    for s, y in bold.items():
        print(f"{s}: {y.shape[0]} TRs x {y.shape[1]} parcels (encoder {a.model}, held-out mean r "
              f"{np.nanmean(enc[s].r_heldout):.3f})")


if __name__ == "__main__":
    main()
