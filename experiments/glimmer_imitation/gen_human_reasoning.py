"""The model's own perception + reasoning for human turns, the input of train_sft.py --human-reasoning.

For human turns (build_sft.py data; per subject half switch / half repeat turns, train and a validation sample), the
model sees the usual agent conversation (frames of the last K turns + current frames + prompt; never the human's
button) and writes its perception and reasoning; generation is forced to start with "<perception>\\n" and stops at
"</reasoning>", so no answer is written. train_sft.py --human-reasoning then trains [this reasoning, no loss] ->
`<answer>human button</answer>` (loss): the human's button is learned after reasoning.
vLLM (vllm_engine.py): one server per GPU pair; resumable (skips turns already in --out).
Output: jsonl, one line per turn: {split, kind, file, i, text}.

  python gen_human_reasoning.py --lora <LoRA .pt> --data-dirs <dir>/sub-01 <dir>/sub-02 --out <jsonl>
  (experiments/glimmer_imitation/job_gen_reasoning.sbatch runs it on 4 GPUs)
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vllm_engine as V  # noqa: E402

ROOT = os.environ.get("MJG_ROOT", ".")
SCRATCH = os.environ.get("MJG_SCRATCH", os.path.join(ROOT, "scratch"))
TMP = os.environ.get("MJG_TMP", os.path.join(SCRATCH, "tmp"))
PREFILL = "<perception>\n"


def say(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def turns_to_write(a, rng):
    """[(split, kind, dataset, play file, turn index)]: per subject an equal share of switch and repeat turns."""
    from train_sft import Turns
    todo = []
    for split, n, mp in (("train", a.n_train, a.max_plays), ("val", a.n_val, a.max_val_plays or a.max_plays)):
        # same plays as train_sft.py with the same --data-dirs / --max-plays (Turns picks them with seed 0)
        ds = Turns(split, mp, a.data_dirs) if a.data_dirs else Turns(split, mp)
        per = n // (2 * len(ds.subjects))  # equal share per subject and per switch / repeat
        for s_ in ds.subjects:
            for kind in ("switch", "repeat"):
                pool = ds.pools[s_][kind]
                for f, i in rng.sample(pool, min(per, len(pool))):
                    todo.append((split, kind, ds, f, i))
    return todo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--gpus", nargs="+", default=["0,1", "2,3"], help="one vLLM server (TP 2) per GPU pair")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dirs", nargs="+", default=None,
                    help="build_sft.py folders, one per subject (default: train_sft.py's default folder)")
    ap.add_argument("--max-plays", type=int, default=0, help="as train_sft.py: plays per folder (train)")
    ap.add_argument("--max-val-plays", type=int, default=0, help="as train_sft.py: plays per folder (val)")
    a = ap.parse_args()
    work = f"{TMP}/genreason_{os.environ.get('SLURM_JOB_ID', 'x')}"
    srvs = []
    for j, g in enumerate(a.gpus):
        srvs.append(V.Client(g, 2, f"{work}/s{j}", port=6001 + j, log=f"{work}_server{j}.log"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # this process only builds prompts
    from glimmer_play import MODEL_DIR, message_images
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    rng = random.Random(0)
    todo = turns_to_write(a, rng)
    done = set()
    if os.path.exists(a.out):
        for r in map(json.loads, open(a.out)):
            done.add((r["split"], r["file"], r["i"]))
    left = []
    for t in todo:
        if (t[0], t[3], t[4]) not in done:
            left.append(t)
    todo = sorted(left, key=lambda t: (t[0], t[3], t[4]))
    say(f"{len(todo)} turns to write ({len(done)} already done)")
    for s in srvs:
        say(f"server ready ({s.wait():.0f}s)")
        s.set_lora(a.lora)
    sp = dict(temperature=1.0, top_p=0.95, top_k=64, max_tokens=a.max_new, stop=["</reasoning>"],
              include_stop_str_in_output=True)
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(len(srvs))
    with open(a.out, "a") as fo:
        for c0 in range(0, len(todo), a.chunk * len(srvs)):
            chunk = todo[c0:c0 + a.chunk * len(srvs)]
            prompts = []
            for split, kind, ds, f, i in chunk:
                msgs = ds.conversation(f, i, 8, "low")[0]
                prompts.append((proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + PREFILL,
                                message_images(msgs)))
            parts = []  # server j gets every len(srvs)-th prompt, from j
            for j in range(len(srvs)):
                parts.append(list(range(j, len(chunk), len(srvs))))
            futs = []
            for j, (srv, idx) in enumerate(zip(srvs, parts)):
                mine = []
                for k in idx:
                    mine.append(prompts[k])
                futs.append(pool.submit(srv.generate, mine, seed=c0 + j, **sp))
            results = []
            for fut in futs:
                results.append(fut.result())
            n_ok = 0
            for (out, _), idx in zip(results, parts):
                for (ids, text, _, _), k in zip(out, idx):
                    split, kind, ds, f, i = chunk[k]
                    if "</reasoning>" not in text:
                        continue  # cut at max_new: not a complete reasoning
                    fo.write(json.dumps(dict(split=split, kind=kind, file=f, i=i, text=PREFILL + text)) + "\n")
                    n_ok += 1
            fo.flush()
            say(f"{c0 + len(chunk)}/{len(todo)} turns, {n_ok}/{len(chunk)} complete in this chunk")
    for s in srvs:
        s.close()
    say("done")


if __name__ == "__main__":
    main()
