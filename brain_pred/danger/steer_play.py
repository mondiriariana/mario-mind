"""Danger steering (GPU): the agent plays the danger scenarios with its danger direction added to its residual stream.

Wraps experiments/glimmer_imitation/glimmer_play.py unchanged (same loading, prompt, sampling, scenario replay,
--save-hidden): only the model loader is wrapped so that, right after loading, a forward hook on decoder layer
(depth d of 13/26/39/52 -> index len(layers) * (d + 1) // 4 - 1, as --save-hidden and embed_scans.py) adds
    alpha * sd * u
to that layer's output at every position (prompt and generated tokens), where u = the unit danger direction from
danger_direction.py (results/danger/direction/<sub>_agent_<tag>_d<d>_danger_dir.npz, key --dir-kind) and sd = the
spread of the training states' projection on u (so alpha is in standard deviations of the state's natural variation
along the direction). alpha 0 = the unsteered agent through the same code. The hook is registered before glimmer_play's
--save-hidden hooks, so the saved states are the steered ones.

  $MJG_MUSE_PY steer_play.py --dir <danger direction npz> --alpha 4 [--dir-kind pattern] -- <glimmer_play args>
(glimmer_play args: --scenarios ... --model-dir <model dir> --lora <SFT lora.pt> --out ...)
"""
import argparse, os, sys

import numpy as np

CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "experiments", "glimmer_imitation"))


def other_runner_alive(lock, me):
    """True if the LOCK file names another Slurm job that is still in the queue."""
    other = open(lock).read().strip()
    if not other or other == me:
        return False
    import subprocess
    return bool(subprocess.run(["squeue", "-h", "-j", other], capture_output=True, text=True).stdout.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--dir-kind", default="pattern", choices=["pattern", "filter", "diffmean"])
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
    z = np.load(a.dir)
    u, sd, depth = z[a.dir_kind].astype(np.float32), float(z[f"sd_{a.dir_kind}"]), int(z["depth"])
    print(f"steering: {a.dir_kind} direction of {os.path.basename(a.dir)}, depth {depth}, "
          f"alpha {a.alpha} x sd {sd:.3f} "
          f"(mean state norm {float(z['mean_state_norm']):.1f})", flush=True)

    # one runner per strength folder: two runners writing the same scenarios would corrupt each other's files.
    # LOCK holds the Slurm job id of the runner; a later runner exits if that job is still alive.
    out = rest[rest.index("--out") + 1]
    os.makedirs(out, exist_ok=True)
    lock, me = os.path.join(out, "LOCK"), os.environ.get("SLURM_JOB_ID", "")
    if os.path.exists(lock) and other_runner_alive(lock, me):
        print(f"{out} is being played by job {open(lock).read().strip()}; exiting", flush=True)
        return
    open(lock, "w").write(me)

    import torch, transformers
    from mjg.lora import find_layers
    Base = transformers.AutoModelForImageTextToText

    class Steered:
        @staticmethod
        def from_pretrained(*args, **kw):
            model = Base.from_pretrained(*args, **kw)
            _, layers = find_layers(model, model.config.text_config.num_hidden_layers)
            li = len(layers) * (depth + 1) // 4 - 1
            assert u.shape[0] == model.config.text_config.hidden_size, (u.shape, model.config.text_config.hidden_size)
            if a.alpha != 0:
                vecs = {}

                def hook(mod, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    if o.device not in vecs:
                        vecs[o.device] = torch.tensor(a.alpha * sd * u, device=o.device, dtype=o.dtype)
                    o = o + vecs[o.device]
                    return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
                layers[li].register_forward_hook(hook)
            print(f"steering hook on decoder layer index {li} of {len(layers)} (alpha {a.alpha})", flush=True)
            return model

    transformers.AutoModelForImageTextToText = Steered
    import glimmer_play
    sys.argv = ["glimmer_play.py"] + rest
    glimmer_play.main()


if __name__ == "__main__":
    main()
