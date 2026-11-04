"""SFT of Glimmer on the human's actions, in the agent's own conversation format. RL (rl.py) starts from this LoRA.

Each sample is one human turn from build_sft.py, in the agent's format (glimmer_play.build_conversation): the
frames of the previous K turns and of the current turn (no past answers), the prompt, and the target
`<answer>['right', 'a']</answer>`. Loss on the target only.

Several subjects (--data-dirs, one folder per subject): the prompt names the player ("Play like player sub-03";
demonstrator-identity conditioning, e.g. ILEED, Beliaev et al. 2022; Maia-Individual, McIlroy-Young et al. 2022),
and every batch and the validation set hold an equal share of each subject.

Copycat guard: humans mostly repeat their previous action, so past answers are not shown, --switch-frac of every
batch comes from turns where the human changed action, and validation reports exact-match separately on switch and
repeat turns (copying the last action scores 0 on switch turns, 1 on repeat). LoRA r16 on every linear layer of the
language model (same LoRA as experiments/glimmer_vlajepa).

Validation (val plays): target loss, and exact-match of the answer under teacher forcing (argmax tokens).

Own reasoning (--human-reasoning, from gen_human_reasoning.py): each human turn becomes [the model's own perception
and reasoning, no loss] -> `<answer>human button</answer>` (loss); turns without a reasoning are dropped.

Reasoning traces (--traces): audited Odysseus turns (one screenshot each, as the agent's current screen; with
--trace-history, the screenshots of earlier turns too). Target = their <perception> and <reasoning> text, then their
<answer>; the loss is on the perception and reasoning only (Odysseus' button choice is never trained). --trace-frac
sets their share of every batch (1 = traces only; validation then also reports the trace loss). --init-lora starts
from a saved LoRA; --model-dir trains on another Glimmer (e.g. the VLA-JEPA one).

Close calls (--close-call-frac): a share of every batch's human turns comes from turns where Mario was at risk
(brain_pred/danger risk model) and the human survived.

Output in results/glimmer_imitation/sft/<tag>/: lora_step<N>.pt at every validation, log.json (validation curve),
train_state.pt (resumable: rerun the same command to continue).

  python train_sft.py --model-dir <VLA-JEPA Glimmer> --data-dirs <build_sft.py out>/sub-01 <...>/sub-02 \\
      --human-reasoning <gen_human_reasoning.py jsonl> --traces <audited Odysseus traces> --trace-frac 0.3125 --tag sft
  (1 H200, one bf16 copy, or 2x L40S, split; experiments/glimmer_imitation/job_sft.sbatch runs it with sft.sh)
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

T_START = time.time()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from glimmer_play import build_conversation, answer_text, message_images, MODEL_DIR  # noqa: E402
from mjg.lora import add_lora, find_layers  # noqa: E402

ROOT = os.environ.get("MJG_ROOT", ".")
SCRATCH = os.environ.get("MJG_SCRATCH", os.path.join(ROOT, "scratch"))
SFT_DATA = f"{SCRATCH}/glimmer_sft/v1"
OUT = f"{ROOT}/results/glimmer_imitation/sft"


def say(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def play_files(d, split):
    """The .npz files of one split of a build_sft.py folder, sorted (plays with at least one turn)."""
    fs = []
    for line in open(f"{d}/index.jsonl"):
        r = json.loads(line)
        if r["split"] == split and r.get("n_turns"):
            fs.append(f"{d}/{split}/{r['play_id']}.npz")
    return sorted(fs)


def kept(items, reasoning):
    out = []
    for x in items:
        if x in reasoning:
            out.append(x)
    return out


def load_rgb(p):
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"))


class Turns:
    """All turns of the plays of one split, pooled over data folders (one per subject); samples are
    (play file, turn index). max_plays caps the plays taken from each folder (random, seed 0)."""
    def __init__(self, split, max_plays=0, dirs=(SFT_DATA,)):
        self.files, self.subject = [], {}
        multi = len(dirs) > 1  # several subjects: name the player in the prompt and balance them
        for d in dirs:
            fs = play_files(d, split)
            if max_plays and len(fs) > max_plays:
                fs = sorted(random.Random(0).sample(fs, max_plays))
            self.files += fs
            for f in fs:
                self.subject[f] = os.path.basename(os.path.normpath(d)) if multi else None
        names = set()
        for s in self.subject.values():
            if s:
                names.add(s)
        self.subjects = sorted(names) or [None]
        self.meta = {}
        self.items, self.switch, self.repeat = [], [], []
        self.pools = {}  # per subject, for balanced batches
        for s in self.subjects:
            self.pools[s] = {"switch": [], "repeat": []}
        for f in self.files:
            z = np.load(f)
            c = z["combo"]
            self.meta[f] = (z["obs_start"], z["obs_len"], c)
            for i in range(len(c)):
                self.items.append((f, i))
                if i > 0:
                    kind = "switch" if c[i] != c[i - 1] else "repeat"
                    if kind == "switch":
                        self.switch.append((f, i))
                    else:
                        self.repeat.append((f, i))
                    self.pools[self.subject[f]][kind].append((f, i))
        self._cache = (None, None)
        self.reasoning = {}

    def frames(self, f):
        if self._cache[0] != f:
            self._cache = (f, np.load(f)["frames"])
        return self._cache[1]

    def conversation(self, f, i, k, strength):
        st, ln, combo = self.meta[f]
        fr = self.frames(f)
        past = []
        for j in range(max(0, i - k), i):
            past.append(turn_frames(fr, st, ln, j))
        msgs = build_conversation(past, turn_frames(fr, st, ln, i), strength, self.subject[f])
        return msgs, self.reasoning.get((f, i), ""), answer_text(int(combo[i])), ""

    def keep_only(self, reasoning):
        """Attach the model's own reasoning (gen_human_reasoning.py) and keep only turns that have one."""
        self.reasoning = reasoning
        self.items = kept(self.items, reasoning)
        self.switch, self.repeat = kept(self.switch, reasoning), kept(self.repeat, reasoning)
        for s_ in self.pools:
            for kind in self.pools[s_]:
                self.pools[s_][kind] = kept(self.pools[s_][kind], reasoning)

    def mark_close_calls(self, risk_dir, thr, horizon=180):
        """Close calls: turns whose risk at their start (brain_pred/danger/risk_model.py: chance Mario dies within
        3 s) is >= thr and where the human then survived the next horizon frames. Pools per subject, from the turns
        kept so far."""
        self.close = {}
        for s in self.subjects:
            self.close[s] = []
        by_file = {}
        for f, i in self.items:
            by_file.setdefault(f, []).append(i)
        for f, idx in by_file.items():
            p = f"{risk_dir}/{os.path.basename(f)}"
            if not os.path.exists(p):
                continue
            rz, t = np.load(p), np.load(f)["t"]
            fin = np.isfinite(rz["risk"])
            fr, rk, deaths = rz["frame"][fin], rz["risk"][fin], rz["death_frame"]
            for i in idx:
                j = np.searchsorted(fr, t[i], side="right") - 1
                if j >= 0 and rk[j] >= thr and not ((deaths > t[i]) & (deaths <= t[i] + horizon)).any():
                    self.close[self.subject[f]].append((f, i))


def turn_frames(fr, st, ln, j):
    """The observation frames of turn j of a play."""
    out = []
    for x in range(st[j], st[j] + ln[j]):
        out.append(fr[x])
    return out


class Traces:
    """Audited Odysseus turns (results/odysseus_sft/v1_audited/<split>.jsonl); items are row indices.
    history > 0: the screenshots of up to `history` earlier turns of the same Odysseus game come first, one frame per
    past turn, as the human examples get their earlier frames (otherwise "earlier frames present" tells the model
    whether to reason or just answer)."""
    def __init__(self, path, history=0):
        self.rows = []
        for line in open(path):
            self.rows.append(json.loads(line))
        self.base = os.path.dirname(path)
        self.items = list(range(len(self.rows)))
        self.history = history

    def conversation(self, f, i, k, strength):
        r = self.rows[i]
        cur = os.path.join(self.base, r["image"])
        past = []
        for t in range(max(1, r["turn"] - min(k, self.history)), r["turn"]):
            p = cur.replace(f"turn-{r['turn']:03d}.png", f"turn-{t:03d}.png")
            if os.path.exists(p):
                past.append([load_rgb(p)])
        msgs = build_conversation(past, [load_rgb(cur)], strength, None)
        sup = f"<perception>\n{r['perception']}\n</perception>\n<reasoning>\n{r['reasoning']}\n</reasoning>\n"
        return msgs, "", sup, f"<answer>{r['buttons']}</answer>"


def reasoning_of(rows, split):
    """gen_human_reasoning.py rows of one split -> {(play file, turn index): reasoning text}."""
    out = {}
    for r in rows:
        if r["split"] == split:
            out[(r["file"], r["i"])] = r["text"] + "\n"
    return out


def val_sample(va, kind, per, rng):
    """Up to per validation turns of this kind (switch / repeat) from every subject."""
    out = []
    for s_ in va.subjects:
        pool = va.pools[s_][kind]
        out.extend(rng.sample(pool, min(per, len(pool))))
    return out


def lora_params(model):
    params = []
    for nm, p in model.named_parameters():
        if nm.endswith((".A", ".B")):
            params.append(p)
    return params


def lora_state(model):
    """The LoRA tensors, on the CPU (what lora_step<N>.pt holds)."""
    sd = {}
    for nm, p in model.named_parameters():
        if nm.endswith((".A", ".B")):
            sd[nm] = p.detach().cpu()
    return sd


def resume_args(d):
    """The arguments that must match for a run to continue a saved state."""
    out = {}
    for k, v in d.items():
        if k not in ("max_minutes", "save_every", "lock_wait"):
            out[k] = v
    return out


def fmt(v):
    s = (f"val loss {v['val_loss']:.4f} | exact switch {v['exact_switch']:.3f} repeat {v['exact_repeat']:.3f} "
         f"natural {v['exact_natural']:.3f} (copy-last baseline {v['copy_baseline_natural']:.3f})")
    if "trace_val_loss" in v:
        s += f" | trace val loss {v['trace_val_loss']:.4f}"
    return s


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--mb", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--strength", default="low")
    ap.add_argument("--val-n", type=int, default=300, help="half switch turns, half repeat turns")
    ap.add_argument("--switch-frac", type=float, default=0.5, help="share of each batch from switch turns")
    ap.add_argument("--evals", type=int, default=5)
    ap.add_argument("--max-plays", type=int, default=0, help="cap on plays per data folder (train); 0 = all")
    ap.add_argument("--max-val-plays", type=int, default=0, help="cap on validation plays per data folder; 0 = all")
    ap.add_argument("--data-dirs", nargs="+", default=[SFT_DATA], help="build_sft.py outputs to pool (one per subject)")
    ap.add_argument("--tag", default="sft_v1")
    ap.add_argument("--traces", default=None, help="Odysseus traces folder with train/val.jsonl (v1_audited)")
    ap.add_argument("--trace-frac", type=float, default=0.0, help="share of each batch from the traces")
    ap.add_argument("--trace-val-n", type=int, default=60)
    ap.add_argument("--trace-history", type=int, default=0, help="earlier screenshots given to each trace (up to --k)")
    ap.add_argument("--init-lora", default=None, help="start from this saved LoRA (lora_step*.pt)")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="save the state and stop after this long (0 = no limit)")
    ap.add_argument("--save-every", type=int, default=20, help="steps between saved training states")
    ap.add_argument("--lock-wait", type=float, default=70, help="minutes to wait for another active run of this tag")
    ap.add_argument("--close-call-frac", type=float, default=0.0,
                    help="share of each batch's human turns drawn from close calls (Turns.mark_close_calls); 0 = off")
    ap.add_argument("--close-call-risk", type=float, default=0.3, help="risk threshold of a close call")
    ap.add_argument("--risk-dir", default=f"{ROOT}/results/danger/risk")
    ap.add_argument("--human-reasoning", default=None,
                    help="jsonl from gen_human_reasoning.py; human examples become [own reasoning, no loss] -> answer")
    a = ap.parse_args()
    assert (a.trace_frac > 0) == bool(a.traces), "--traces and --trace-frac go together"
    return a


def load_model(a):
    """Processor, Glimmer in bf16 (one GPU if it has > 80 GB, else split over the GPUs) with LoRA on every linear
    layer of the language model, and the LoRA parameters."""
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(MODEL_DIR)  # the original processor (fine-tuning does not change it)
    tok = proc.tokenizer
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if torch.cuda.get_device_properties(0).total_memory > 80e9:
        kw = dict(device_map={"": 0})
    else:
        mem = {}
        for g in range(torch.cuda.device_count()):
            mem[g] = "36GiB" if g == 0 else "44GiB"
        kw = dict(device_map="auto", max_memory=mem)
    model = AutoModelForImageTextToText.from_pretrained(a.model_dir, dtype=torch.bfloat16, **kw)
    model.requires_grad_(False)
    _, layers = find_layers(model, model.config.text_config.num_hidden_layers)
    n = add_lora(layers, a.r)
    if a.init_lora:
        init = torch.load(a.init_lora, map_location="cpu")
        with torch.no_grad():
            for nm, p in model.named_parameters():
                if nm.endswith((".A", ".B")):
                    p.copy_(init.pop(nm).to(p.device, p.dtype))
        assert not init, f"LoRA tensors not used: {list(init)[:3]}"
        say(f"LoRA initialised from {a.init_lora}")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = lora_params(model)
    n_params = 0
    for p in params:
        n_params += p.numel()
    say(f"LoRA {n} modules, {n_params / 1e6:.1f}M params")
    return proc, model, params


def main():
    # On Hopper (H200) PyTorch's cuDNN attention backward gives NaN gradients in Glimmer (see
    # experiments/glimmer_vlajepa/train.py); with it off SDPA uses FlashAttention, as on the L40S.
    torch.backends.cuda.enable_cudnn_sdp(False)
    a = parse_args()
    od = f"{OUT}/{a.tag}"
    os.makedirs(od, exist_ok=True)
    random.seed(0)
    torch.manual_seed(0)

    tr = Turns("train", a.max_plays, a.data_dirs)
    va = Turns("val", a.max_val_plays or a.max_plays, a.data_dirs)
    rng = random.Random(0)
    if a.human_reasoning:
        rs = []
        for line in open(a.human_reasoning):
            rs.append(json.loads(line))
        tr.keep_only(reasoning_of(rs, "train"))
        va.keep_only(reasoning_of(rs, "val"))
    if a.close_call_frac:
        tr.mark_close_calls(a.risk_dir, a.close_call_risk)
        n_close = {}
        for s_, v in tr.close.items():
            n_close[s_] = len(v)
        say("close calls (risk >= {:g}, survived 3 s): {}; {:.2f} of every batch's human turns".format(
            a.close_call_risk, n_close, a.close_call_frac))
        assert all(tr.close.values()), "a subject has no close calls"
    per = max(1, a.val_n // (2 * len(va.subjects)))  # equal validation turns per subject
    val_sw = val_sample(va, "switch", per, rng)
    val_rp = val_sample(va, "repeat", per, rng)
    p_switch = len(va.switch) / max(1, len(va.switch) + len(va.repeat))
    say(f"train {len(tr.items)} turns ({len(tr.files)} plays; {len(tr.switch)} switch / {len(tr.repeat)} repeat), "
        f"val {len(va.items)} turns (switch share {p_switch:.3f}); val sets {len(val_sw)} switch + {len(val_rp)} "
        f"repeat; subjects {tr.subjects} (batches balanced across subjects)")
    if a.traces:
        ttr = Traces(f"{a.traces}/train.jsonl", a.trace_history)
        tva = Traces(f"{a.traces}/val.jsonl", a.trace_history)
        val_tr = []
        for i in rng.sample(tva.items, min(a.trace_val_n, len(tva.items))):
            val_tr.append((None, i))
        say(f"traces: {len(ttr.items)} train, {len(val_tr)} val; {a.trace_frac:.2f} of every batch")

    proc, model, params = load_model(a)
    tok = proc.tokenizer
    dev = model.get_input_embeddings().weight.device

    def encode(items):
        """items: (dataset, play file, index) -> batch dict on dev; labels on the supervised target only
        (the action answer + EOS for human turns; perception + reasoning for traces, not their answer)."""
        full, prompts, sups = [], [], []
        for ds, f, i in items:
            msgs, pre, sup, tail = ds.conversation(f, i, a.k, a.strength)
            p = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + pre
            prompts.append((msgs, p))
            sups.append((sup, tail))
            full.append(p + sup + tail + tok.eos_token)
        imgs = []
        for msgs, _ in prompts:
            imgs.append(message_images(msgs))
        b = proc(text=full, images=imgs, return_tensors="pt", padding=True)
        labels = b["input_ids"].clone()
        labels[b["attention_mask"] == 0] = -100
        for j, ((msgs, p), im, (sup, tail)) in enumerate(zip(prompts, imgs, sups)):
            npr = proc(text=[p], images=[im], return_tensors="pt")["input_ids"].shape[1]
            labels[j, :npr] = -100
            if tail:
                labels[j, proc(text=[p + sup], images=[im], return_tensors="pt")["input_ids"].shape[1]:] = -100
        b["labels"] = labels
        out = {}
        for key, v in b.items():
            out[key] = v.to(dev) if torch.is_tensor(v) else v
        return out

    def validate():
        """-> dict: val loss, exact-match on switch / repeat turns, and the natural-mix accuracy
        (switch share of the val data); copying the last action would score 0 / 1 / 1 - p_switch."""
        model.eval()
        tot, cnt, ex = 0.0, 0, {"switch": 0, "repeat": 0}
        with torch.no_grad():
            for name, items in (("switch", val_sw), ("repeat", val_rp)):
                for it in items:
                    b = encode([(va, *it)])
                    out = model(**b)
                    lab = b["labels"][0, 1:]
                    m = lab != -100
                    pred = out.logits[0, :-1].argmax(-1)
                    tot += float(out.loss) * int(m.sum())
                    cnt += int(m.sum())
                    ex[name] += int((pred[m] == lab[m]).all())
            ttot, tcnt = 0.0, 0
            for it in (val_tr if a.traces else []):
                b = encode([(tva, *it)])
                m = int((b["labels"][0, 1:] != -100).sum())
                ttot += float(model(**b).loss) * m
                tcnt += m
        model.train()
        es, er = ex["switch"] / max(1, len(val_sw)), ex["repeat"] / max(1, len(val_rp))
        res = dict(val_loss=tot / cnt, exact_switch=es, exact_repeat=er,
                   exact_natural=p_switch * es + (1 - p_switch) * er, copy_baseline_natural=1 - p_switch)
        if tcnt:
            res["trace_val_loss"] = ttot / tcnt
        return res

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warm = max(1, a.steps // 30)

    def lr_factor(s):  # linear warm-up, then cosine decay
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    # Resumable in chunks (e.g. time-limited sessions): the state (LoRA, optimizer, schedule, sampler RNG, log) is
    # saved every --save-every steps and when --max-minutes is reached; a run with the same arguments (whatever its
    # --max-minutes / --save-every) continues from it. A lock refreshed at every save stops two runs of one tag.
    state_path, lock = f"{od}/train_state.pt", f"{od}/RUNNING"

    def lock_fresh():
        return os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 25 * 60
    if lock_fresh():  # another run of this tag is active: wait for it to pause, then take over
        say(f"{lock} is fresh: waiting up to {a.lock_wait} min for the other run to pause")
        t_w = time.time()
        while lock_fresh() and time.time() - t_w < a.lock_wait * 60:
            time.sleep(60)
        if lock_fresh():
            say("still busy; exiting")
            return
    start = 0
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        assert resume_args(st["args"]) == resume_args(vars(a)), "train_state.pt was written with other arguments"
        with torch.no_grad():
            for nm, p_ in model.named_parameters():
                if nm in st["lora"]:
                    p_.copy_(st["lora"][nm].to(p_.device))
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        rng.setstate(st["rng"])
        start, log = st["step"], st["log"]
        say(f"resumed from {state_path} at step {start}/{a.steps}")
        if start >= a.steps:
            say("already done")
            return
    else:
        v0 = validate()
        log = dict(args=vars(a), curve=[dict(step=0, **v0)])
        say(f"step 0 {fmt(v0)}")

    def save_state(step):
        torch.save(dict(step=step, args=vars(a), log=log, rng=rng.getstate(), opt=opt.state_dict(),
                        sched=sched.state_dict(), lora=lora_state(model)),
                   state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
        open(lock, "w").write(os.environ.get("SLURM_JOB_ID", "local"))
    open(lock, "w").write(os.environ.get("SLURM_JOB_ID", "local"))
    every = max(1, a.steps // a.evals)
    model.train()
    t0 = time.time()
    for s in range(start + 1, a.steps + 1):
        n_tr = round(a.bs * a.trace_frac)
        n_act = a.bs - n_tr
        n_sw = round(n_act * a.switch_frac)
        subj = []  # equal share of every subject
        for j in range(n_act):
            subj.append(tr.subjects[j % len(tr.subjects)])
        rng.shuffle(subj)
        batch = []
        for j, s_ in enumerate(subj):
            batch.append((tr, *rng.choice(tr.pools[s_]["switch" if j < n_sw else "repeat"])))
        if a.close_call_frac:  # replace that share of the human turns (spread over subjects) with close calls
            for j in rng.sample(range(n_act), round(n_act * a.close_call_frac)):
                batch[j] = (tr, *rng.choice(tr.close[subj[j]]))
        for _ in range(n_tr):
            batch.append((ttr, None, rng.choice(ttr.items)))
        rng.shuffle(batch)
        for j in range(0, a.bs, a.mb):
            loss = model(**encode(batch[j:j + a.mb])).loss * (min(a.mb, a.bs - j) / a.bs)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if s % 10 == 0:
            sec = (time.time() - t0) / (s - start)
            say(f"step {s}/{a.steps} loss {float(loss) * a.bs / a.mb:.4f} ({sec:.1f} s/step)")
        if s % every == 0 or s == a.steps:
            v = validate()
            log["curve"].append(dict(step=s, sec=round(time.time() - t0), **v))
            say(f"step {s} {fmt(v)}")
            torch.save(lora_state(model), f"{od}/lora_step{s}.pt")
            json.dump(log, open(f"{od}/log.json", "w"), indent=1)
        if s % a.save_every == 0 or s == a.steps:
            save_state(s)
        if a.max_minutes and time.time() - T_START > a.max_minutes * 60 and s < a.steps:
            save_state(s)
            os.remove(lock)
            say(f"paused at step {s}/{a.steps} (--max-minutes); rerun the same command to continue")
            return
    os.remove(lock)
    say("done")


if __name__ == "__main__":
    main()
