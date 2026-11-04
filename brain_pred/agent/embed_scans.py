"""Calibration data, part 2 (GPU): the agent's internal state at every fMRI scan of one subject.

For each scan (TR) of each run, the play on screen and its frame window come from mjg.encoding.scan_table (as for
every other feature band). The agent's state at that moment = its state at its most recent decision: the last turn
of the participant's play (build_scan_turns.py) starting before the end of the scan. The agent gets exactly the
conversation it would have if it were playing there (glimmer_play.build_conversation: the frames of its last K
turns + the current turn's frames + the prompt; nothing about the human's action), one forward pass (no generation),
and we keep the hidden state of the last prompt token at 4 depths (decoder layers 13/26/39/52), the same quantity
glimmer_play --save-hidden records while the agent plays. That is what lets an encoder fitted here generate BOLD from
the agent's own play.
Writes $MJG_ROOT/results/brain_encoding/tr_features/<sub>/<run>.agent_<tag>.npy float16 [n_TR, 4, hidden] and
<run>.agent_<tag>.mask.npy (scans with an agent state). Resumable per run; --part i/N splits the runs.

  python embed_scans.py --subject sub-03 --tag sft --lora <SFT lora.pt> --model-dir <VLA-JEPA Glimmer> --player sub-03
  python embed_scans.py --subject sub-03 --tag plain --part 0/4                 (no LoRA: plain Glimmer)
--reason (4 GPUs): the agent also WRITES its reply at each decision (vLLM on GPUs 0,1, same prompt and sampling as
when it plays), then one forward pass reads prompt + reply and keeps two more bands at the same 4 depths:
<tag>reason = mean hidden state over the reply's tokens (its reasoning, then its answer), <tag>post = the state of the
reply's last token. The replies go to <run>.agent_<tag>.replies.jsonl (one line per decision). Nothing about the human.
"""
import argparse, json, os, re, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "glimmer_imitation"))
from mjg import encoding as E  # noqa: E402
from glimmer_play import build_conversation, chunk_vision, MODEL_DIR  # noqa: E402

ROOT = os.environ.get("MJG_ROOT", ".")
SCRATCH = os.environ.get("MJG_SCRATCH", os.path.join(ROOT, "scratch"))
TMP = os.environ.get("MJG_TMP", os.path.join(SCRATCH, "tmp"))
TURNS = f"{SCRATCH}/glimmer_scan"
TRF = f"{ROOT}/results/brain_encoding/tr_features"
LOCK_STALE_S = 3 * 3600  # a lock older than this belongs to a worker that died


def say(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def play_id(stem):  # as brain_pred/encoding/build_tr_features.play_id
    m = re.match(r"(sub-\d+)_(ses-\d+)_task-mario_level-(w\dl\d)_rep-(\d+)", stem)
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}_rep-{m.group(4)}" if m else stem


def images_of(msgs):
    """The images of a conversation, in order."""
    out = []
    for x in msgs:
        content = x["content"] if isinstance(x["content"], list) else []
        for c in content:
            if c.get("type") == "image":
                out.append(c["image"])
    return out


def obs_frames(z, j):
    """The observation frames of turn j of a scan-turns file."""
    out = []
    for x in range(z["obs_start"][j], z["obs_start"][j] + z["obs_len"][j]):
        out.append(z["frames"][x])
    return out


def run_seed(key):
    """Sampling seed of a run: the same replies every time it is embedded."""
    s = 0
    for ch in key:
        s += ord(ch)
    return s


def device_kw(torch):
    """Load on GPU 0 when it holds the model (>= 80 GB), else shard over every visible GPU."""
    if torch.cuda.get_device_properties(0).total_memory >= 80e9:
        return dict(device_map={"": 0})
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        max_memory[i] = "36GiB" if i == 0 else "44GiB"
    return dict(device_map="auto", max_memory=max_memory)


def select_runs(a, keys, i_part, n_part):
    """This part's runs: a share of them, or with --pool all of them from a part-specific starting point."""
    if a.heldout_only:  # only the held-out sessions' runs
        hold = set(json.load(open(os.path.join(REPO, "configs", "holdout_sessions.json")))["holdout"][a.subject])
        held = []
        for k in keys:
            if k.split("_")[0] in hold:
                held.append(k)
        keys = held
    if not a.pool:
        return keys[i_part::n_part]
    if i_part:  # pool: the parts start at different places, so they rarely race for the same lock
        cut = len(keys) * i_part // n_part
        return keys[cut:] + keys[:cut]
    return keys


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="sub-03")
    ap.add_argument("--tag", required=True, help="name of the agent, e.g. sft / plain")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--player", default=None, help="player tag for multi-subject agents (sub-0X)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--turns-split", default="all", help="all = turns until the first death; all_lives = through "
                    "every life (build_scan_turns.py --all-lives), so later lives get fresh states")
    ap.add_argument("--strength", default="low")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--reason", action="store_true", help="also generate the reply and embed it (see docstring)")
    ap.add_argument("--replies-only", action="store_true", help="with --reason: write the replies and the prompt "
                    "states but skip the forward pass over prompt + reply (<tag>reason / <tag>post). The run is done "
                    "when <run>.agent_<tag>.replies.jsonl exists (written last)")
    ap.add_argument("--no-state", action="store_true", help="with --reason --replies-only: only the replies (vLLM on "
                    "2 GPUs, no HF model), so a 4-GPU node runs two of these; the prompt states come in a later pass "
                    "(same tag, without --reason)")
    ap.add_argument("--pool", action="store_true", help="every part takes any run not done and not locked "
                    "(<run>.agent_<tag>.lock, stale after 3 h) instead of its own share: several jobs / windows / "
                    "nodes can work on one agent at once and none redoes another's run")
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--max-runs", type=int, default=0, help="only the first N runs of this part")
    ap.add_argument("--heldout-only", action="store_true", help="only the runs of the held-out sessions "
                    "(configs/holdout_sessions.json)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="chat template with enable_thinking=False (Qwen3.5: no long hidden thinking; as in its play)")
    return ap.parse_args()


def main():
    a = parse_args()
    part = a.part.split("/")
    i_part, n_part = int(part[0]), int(part[1])
    srv = None
    if a.reason:  # the vLLM server takes GPUs 0,1 before this process touches CUDA; the HF model gets the rest
        import vllm_engine
        gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
        work = os.path.join(TMP, f"vllm_embed_{a.tag}_{os.environ.get('SLURM_JOB_ID', 'x')}_{i_part}")
        srv = vllm_engine.Client(",".join(gpus[:2]), 2, work, port=6001 + i_part, model=a.model_dir)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus[2:])
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from mjg.lora import add_lora, find_layers

    n_trs = E.run_trs_from_confounds(a.subject)
    st = E.scan_table(a.subject, n_trs)
    keys = select_runs(a, sorted(st.key.unique()), i_part, n_part)
    last_tag = f"{a.tag}post" if a.reason else a.tag  # written last: a run is done when it exists

    def done_path(k):
        if a.reason and a.replies_only:
            return f"{TRF}/{a.subject}/{k}.agent_{a.tag}.replies.jsonl"
        return f"{TRF}/{a.subject}/{k}.agent_{last_tag}.npy"

    def lock_path(k):  # one lock per pass
        return f"{TRF}/{a.subject}/{k}.agent_{a.tag}{'.r' if a.reason else ''}.lock"

    todo = []
    for k in keys:
        if not os.path.exists(done_path(k)):
            todo.append(k)
    if a.max_runs:
        todo = todo[:a.max_runs]
    say(f"{a.subject}: part {a.part}: {len(keys)} runs, {len(todo)} to do")
    if not todo:
        return

    # other models (Qwen3.5, Odysseus, Llama 3.2 Vision) bring their own processor; a fine-tuned Glimmer folder may not
    has_proc = os.path.exists(f"{a.model_dir}/tokenizer_config.json")
    proc = AutoProcessor.from_pretrained(a.model_dir if has_proc else MODEL_DIR)
    chat_kw = dict(enable_thinking=False) if a.no_thinking else {}
    proc.tokenizer.padding_side = "left"
    if proc.tokenizer.pad_token_id is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    no_state = a.no_state and a.reason and a.replies_only
    if no_state:  # replies only: no HF model on this process's GPUs
        layers, model = [None] * 4, None
    else:
        model = AutoModelForImageTextToText.from_pretrained(a.model_dir, dtype=torch.bfloat16,
                                                            **device_kw(torch)).eval()
        chunk_vision(model)
        _, layers = find_layers(model, model.config.text_config.num_hidden_layers)
    if a.lora and not no_state:
        add_lora(layers, 16)
        sd = torch.load(a.lora, map_location="cpu")
        params = dict(model.named_parameters())
        with torch.no_grad():
            for nm, v in sd.items():
                params[nm].copy_(v.to(params[nm].device))
        say(f"LoRA {a.lora}")
    if srv:
        say(f"vLLM server ready ({srv.wait():.0f}s)")
        if a.lora:
            srv.set_lora(a.lora)
    depth = []
    for q in (1, 2, 3, 4):
        depth.append(len(layers) * q // 4 - 1)
    cap, span = {}, [0]

    def hook(i):
        def f(mod, inp_, out):
            o = out[0] if isinstance(out, tuple) else out
            cap[i] = o[:, -1].detach().float().cpu()
            if span[0]:  # --reason: the last span[0] positions (the replies, left padding keeps them at the end)
                cap[("span", i)] = o[:, -span[0]:].detach().float().cpu()
        return f
    if not no_state:
        for i in depth:
            layers[i].register_forward_hook(hook(i))
    H = model.config.text_config.hidden_size if model is not None else 1

    cache_play = {}

    def turns_of(pid):
        if pid not in cache_play:
            p = f"{TURNS}/{a.subject}/{a.turns_split}/{pid}.npz"
            cache_play.clear()
            cache_play[pid] = dict(np.load(p)) if os.path.exists(p) else None
        return cache_play[pid]

    def conv(z, i):
        past = list(range(max(0, i - a.k), i))
        # all-lives turns: the history restarts at each life, as when the agent plays (a death ends its episode)
        if "life" in z:
            same_life = []
            for j in past:
                if z["life"][j] == z["life"][i]:
                    same_life.append(j)
            past = same_life
        past_frames = []
        for j in past:
            past_frames.append(obs_frames(z, j))
        return build_conversation(past_frames, obs_frames(z, i), a.strength, a.player)

    def prompts(msgs):
        texts, images = [], []
        for m in msgs:
            texts.append(proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **chat_kw))
            images.append(images_of(m))
        return texts, images

    def to_device(inp):
        out = {}
        for k, v in inp.items():
            out[k] = v.to(model.device) if torch.is_tensor(v) else v
        return out

    def stack_depths(key_of):
        per_depth = []
        for i in depth:
            per_depth.append(key_of(i))
        return torch.stack(per_depth, 1)

    @torch.no_grad()
    def embed(msgs):
        texts, images = prompts(msgs)
        inp = to_device(proc(text=texts, images=images, return_tensors="pt", padding=True))
        inp.pop("token_type_ids", None)
        model.model(**inp, use_cache=False)
        return stack_depths(lambda i: cap[i]).half().numpy()  # [B, 4, H]

    @torch.no_grad()
    def embed_reply(msgs, gens):
        """prompt + the agent's own reply (token ids from vLLM): mean state over the reply, state of its last token."""
        texts, images = prompts(msgs)
        inp = proc(text=texts, images=images, return_tensors="pt", padding=True)
        inp.pop("token_type_ids", None)
        ids, am = inp["input_ids"], inp["attention_mask"]
        rows = []
        for b, g in enumerate(gens):
            rows.append(torch.cat([ids[b][am[b] == 1], torch.tensor(g, dtype=ids.dtype)]))
        L = 0
        for r in rows:
            L = max(L, len(r))
        padded, masks = [], []
        for r in rows:
            padded.append(torch.cat([torch.full((L - len(r),), proc.tokenizer.pad_token_id, dtype=ids.dtype), r]))
            masks.append(torch.cat([torch.zeros(L - len(r), dtype=am.dtype), torch.ones(len(r), dtype=am.dtype)]))
        inp["input_ids"], inp["attention_mask"] = torch.stack(padded), torch.stack(masks)
        inp = to_device(inp)
        longest = 0
        for g in gens:
            longest = max(longest, len(g))
        span[0] = longest
        try:
            model.model(**inp, use_cache=False)
        finally:
            span[0] = 0

        def reply_mean(i):
            per_reply = []
            for b, g in enumerate(gens):
                per_reply.append(cap[("span", i)][b, -len(g):].mean(0))
            return torch.stack(per_reply)
        mean = stack_depths(reply_mean)
        post = stack_depths(lambda i: cap[i])
        return mean.half().numpy(), post.half().numpy()  # [B, 4, H] each

    def take(key):  # --pool: claim a run with an exclusive lock file; False if done or another worker has it
        if os.path.exists(done_path(key)):
            return False
        if not a.pool:
            return True
        lk = lock_path(key)
        if os.path.exists(lk) and time.time() - os.path.getmtime(lk) > LOCK_STALE_S:
            os.remove(lk)  # its worker died (window cut, node reboot)
        try:
            os.close(os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return True
        except FileExistsError:
            return False

    t0, done = time.time(), 0
    for key in todo:
        if not take(key):
            continue
        g = st[st.key == key].sort_values("tr")
        T = len(g)
        out = np.zeros((T, 4, H), np.float16)
        mask = np.zeros(T, bool)
        items = []  # (tr index, play id, turn index)
        for r in g.itertuples():
            if not r.stem or r.f1 <= 0:
                continue
            pid = play_id(r.stem)
            z = turns_of(pid)
            if z is None or not len(z["t"]):
                continue
            i = int(np.searchsorted(z["t"], r.f1, side="left")) - 1  # last turn starting before the scan ends
            if i < 0:
                continue
            items.append((int(r.tr), pid, i))
        turns_used = set()
        for _, p, i in items:
            turns_used.add((p, i))
        uniq = sorted(turns_used)
        vec, vr, vp, gen = {}, {}, {}, {}
        if srv and uniq:  # all of this run's replies in one vLLM call (seeded per run: reproducible)
            msgs = []
            for p, i in uniq:
                msgs.append(conv(turns_of(p), i))
            texts, images = prompts(msgs)
            outs, sec = srv.generate(list(zip(texts, images)), temperature=1.0, top_p=0.95, top_k=64,
                                     max_tokens=a.max_new, seed=run_seed(key))
            n_tokens = []
            for u, o in zip(uniq, outs):
                gen[u] = o
                n_tokens.append(len(o[0]))
            say(f"  {key}: {len(uniq)} replies in {sec:.0f}s, mean {np.mean(n_tokens):.0f} tokens")
        base = f"{TRF}/{a.subject}/{key}.agent_{a.tag}.npy"
        # --reason on an agent already embedded: keep its prompt-state files
        do_base = not os.path.exists(base) and not no_state
        for b in range(0, len(uniq), a.batch):
            chunk = uniq[b:b + a.batch]
            msgs = []
            for p, i in chunk:
                msgs.append(conv(turns_of(p), i))
            if do_base:
                for (p, i), v in zip(chunk, embed(msgs)):
                    vec[(p, i)] = v
            if srv and not a.replies_only:
                gens = []
                for u in chunk:
                    gens.append(gen[u][0])
                for (p, i), r, q in zip(chunk, *embed_reply(msgs, gens)):
                    vr[(p, i)], vp[(p, i)] = r, q
        for tr, p, i in items:
            mask[tr] = True
            if do_base:
                out[tr] = vec[(p, i)]
        os.makedirs(f"{TRF}/{a.subject}", exist_ok=True)
        if do_base:
            np.save(f"{TRF}/{a.subject}/{key}.agent_{a.tag}.mask.npy", mask)
            np.save(base, out)
        if srv:  # written last: <tag>post marks the run as done
            rp = f"{TRF}/{a.subject}/{key}.agent_{a.tag}.replies.jsonl"
            with open(rp + ".tmp", "w") as f:  # renamed when complete: with --replies-only it marks the run as done
                for p, i in uniq:
                    f.write(json.dumps(dict(play=p, turn=int(i), reply=gen[(p, i)][1],
                                            n_tokens=len(gen[(p, i)][0]))) + "\n")
            os.replace(rp + ".tmp", rp)
            for t2, v2 in (() if a.replies_only else ((f"{a.tag}reason", vr), (f"{a.tag}post", vp))):
                o2 = np.zeros_like(out)
                for tr, p, i in items:
                    o2[tr] = v2[(p, i)]
                np.save(f"{TRF}/{a.subject}/{key}.agent_{t2}.mask.npy", mask)
                np.save(f"{TRF}/{a.subject}/{key}.agent_{t2}.npy", o2)
        lk = lock_path(key)
        if a.pool and os.path.exists(lk):
            os.remove(lk)
        done += len(uniq)
        per_state = (time.time() - t0) / max(1, done)
        say(f"  {key}: {mask.sum()}/{T} scans, {len(uniq)} agent states ({per_state:.2f} s each)")


if __name__ == "__main__":
    main()
