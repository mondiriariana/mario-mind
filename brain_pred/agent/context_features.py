"""Linguistic context (GPU): the agent's language model reading its own reasoning so far.

embed_scans.py --reason writes, at each decision, the agent's reply (<run>.agent_<tag>.replies.jsonl: its own
perception, reasoning and answer; nothing about the human). Here the same language model reads, text only and in
order, every reply the agent wrote in the play so far, as a language encoding model reads a story: the state at the
end of each reply carries that reply in the context of the previous ones (causal attention, so only the past). One
long pass per play in windows of --window tokens with stride --window/2: each reply is read in the window that gives
it at least --window/2 tokens of earlier replies (all of them when the play is shorter).
Every scan gets the state of its most recent decision (same rule as embed_scans.py), at the same 4 depths.
Writes $MJG_ROOT/results/brain_encoding/tr_features/<sub>/<run>.agent_<tag>ctx.npy float16 [n_TR, 4, hidden] +
.mask.npy, so encode.py --agent-tags <tag>ctx gives bands agent_<tag>ctx_d0..d3. Resumable per run; --part i/N
splits the runs.

  python context_features.py --subject sub-03 --tag sft --lora <SFT lora.pt> --model-dir <VLA-JEPA Glimmer> --part 0/2
"""
import argparse, json, os, re, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mjg import encoding as E  # noqa: E402

ROOT = os.environ.get("MJG_ROOT", ".")
SCRATCH = os.environ.get("MJG_SCRATCH", os.path.join(ROOT, "scratch"))
MODELS = os.environ.get("MJG_MODELS", os.path.join(ROOT, "models"))
TURNS = f"{SCRATCH}/glimmer_scan"
TRF = f"{ROOT}/results/brain_encoding/tr_features"


def say(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def play_id(stem):  # as brain_pred/agent/embed_scans.py
    m = re.match(r"(sub-\d+)_(ses-\d+)_task-mario_level-(w\dl\d)_rep-(\d+)", stem)
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}_rep-{m.group(4)}" if m else stem


def device_kw(torch):
    """Load on GPU 0 when it holds the model (>= 80 GB), else shard over every visible GPU."""
    if torch.cuda.get_device_properties(0).total_memory >= 80e9:
        return dict(device_map={"": 0})
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        max_memory[i] = "36GiB" if i == 0 else "44GiB"
    return dict(device_map="auto", max_memory=max_memory)


def capture_hook(cap, i):
    """Forward hook: keep layer i's output states of the (single) sequence in cap[i], [L, H]."""
    def hook(mod, inp_, out):
        o = out[0] if isinstance(out, tuple) else out
        cap[i] = o[0].detach()
    return hook


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="sub-03")
    ap.add_argument("--tag", default="sft", help="the agent whose replies.jsonl are read (embed_scans.py --reason)")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--turns-split", default="all", help="the turns the replies were written for")
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--max-runs", type=int, default=0, help="only the first N runs of this part")
    ap.add_argument("--no-answer", action="store_true", help="drop each reply's <answer> (its buttons) from the text: "
                    "control for motor content; writes tag <tag>ctxna")
    a = ap.parse_args()
    part = a.part.split("/")
    i_part, n_part = int(part[0]), int(part[1])
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    from mjg.lora import add_lora, find_layers

    n_trs = E.run_trs_from_confounds(a.subject)
    st = E.scan_table(a.subject, n_trs)
    d = f"{TRF}/{a.subject}"
    keys = []
    for k in sorted(st.key.unique()):
        if os.path.exists(f"{d}/{k}.agent_{a.tag}.replies.jsonl"):
            keys.append(k)
    keys = keys[i_part::n_part]
    out_tag = f"{a.tag}ctx" + ("na" if a.no_answer else "")
    todo = []
    for k in keys:
        if not os.path.exists(f"{d}/{k}.agent_{out_tag}.npy"):
            todo.append(k)
    if a.max_runs:
        todo = todo[:a.max_runs]
    say(f"{a.subject}: part {a.part}: {len(keys)} runs with replies, {len(todo)} to do")
    if not todo:
        return

    has_tok = os.path.exists(f"{a.model_dir}/tokenizer_config.json")
    tok = AutoTokenizer.from_pretrained(a.model_dir if has_tok else f"{MODELS}/Muse-Glimmer-30B")
    model = AutoModelForImageTextToText.from_pretrained(a.model_dir, dtype=torch.bfloat16, **device_kw(torch)).eval()
    _, layers = find_layers(model, model.config.text_config.num_hidden_layers)
    if a.lora:
        add_lora(layers, 16)
        sd = torch.load(a.lora, map_location="cpu")
        params = dict(model.named_parameters())
        with torch.no_grad():
            for nm, v in sd.items():
                params[nm].copy_(v.to(params[nm].device))
        say(f"LoRA {a.lora}")
    depth = []
    for q in (1, 2, 3, 4):
        depth.append(len(layers) * q // 4 - 1)
    cap = {}
    for i in depth:
        layers[i].register_forward_hook(capture_hook(cap, i))
    H = model.config.text_config.hidden_size
    S = a.window // 2
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []

    @torch.no_grad()
    def read_play(replies):
        """replies in order -> [n, 4, H]: the state at the last token of each, with the earlier replies as context."""
        ids, ends = list(bos), []
        for r in replies:
            if a.no_answer:
                r = re.sub(r"<answer>.*?(</answer>|$)", "", r, flags=re.S)
            ids += tok(r.strip() + "\n\n", add_special_tokens=False)["input_ids"]
            ends.append(len(ids) - 1)
        out = np.zeros((len(replies), 4, H), np.float16)
        need = {}  # window start -> reply indices read there
        for j, p in enumerate(ends):
            w0 = 0 if p < a.window else (p // S - 1) * S  # p in [w0 + S, w0 + 2S): >= S tokens of context
            need.setdefault(w0, []).append(j)
        for w0, js in sorted(need.items()):
            x = torch.tensor([ids[w0:w0 + a.window]], device=model.device)
            model.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)  # text only
            offsets = []
            for j in js:
                offsets.append(ends[j] - w0)
            pos = torch.tensor(offsets)
            per_depth = []
            for i in depth:
                per_depth.append(cap[i][pos].float().cpu())
            out[js] = torch.stack(per_depth, 1).half().numpy()
        return out, len(ids)

    t0, n_rep, n_tok = time.time(), 0, 0
    for key in todo:
        rep = []
        for line in open(f"{d}/{key}.agent_{a.tag}.replies.jsonl"):
            rep.append(json.loads(line))
        by_play = {}
        for r in rep:
            by_play.setdefault(r["play"], []).append(r)
        vec = {}
        for pid, rs in by_play.items():
            rs.sort(key=lambda r: r["turn"])
            texts = []
            for r in rs:
                texts.append(r["reply"])
            v, nt = read_play(texts)
            for r, x in zip(rs, v):
                vec[(pid, r["turn"])] = x
            n_tok += nt
        n_rep += len(rep)
        g = st[st.key == key].sort_values("tr")
        T = len(g)
        out = np.zeros((T, 4, H), np.float16)
        mask = np.zeros(T, bool)
        cache = {}
        for r in g.itertuples():
            if not r.stem or r.f1 <= 0:
                continue
            pid = play_id(r.stem)
            if pid not in cache:
                p = f"{TURNS}/{a.subject}/{a.turns_split}/{pid}.npz"
                cache[pid] = np.load(p)["t"] if os.path.exists(p) else None
            tt = cache[pid]
            if tt is None or not len(tt):
                continue
            i = int(np.searchsorted(tt, r.f1, side="left")) - 1  # the decision embed_scans.py used
            if i >= 0 and (pid, i) in vec:
                out[int(r.tr)] = vec[(pid, i)]
                mask[int(r.tr)] = True
        np.save(f"{d}/{key}.agent_{out_tag}.mask.npy", mask)
        np.save(f"{d}/{key}.agent_{out_tag}.npy", out)  # written last: the run is done
        el = time.time() - t0
        say(f"  {key}: {len(rep)} replies in {len(by_play)} plays, {mask.sum()}/{T} scans "
            f"({n_tok / el:.0f} tokens/s, {el / max(1, n_rep):.2f} s per reply)")


if __name__ == "__main__":
    main()
