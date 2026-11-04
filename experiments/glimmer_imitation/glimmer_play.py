"""Glimmer plays Super Mario Bros, turn-based with history. Zero-shot, or with a trained LoRA (--lora).
Reuses the Odysseus episode / emulator code (experiments/odysseus/play.py).

Per turn:
  - the game is paused while Glimmer thinks (as Odysseus and Reason-to-Play)
  - observation = the frames since its last action (every --every-th frame of the hold, plus the last one;
    the last one at 2x, the others at 1x), then the prompt
  - history: the frames of the previous K turns, oldest dropped; one user message, oldest frame first. Past answers
    are not shown: they let a model copy its last action (the copycat problem)
  - answer: a list of up to 3 buttons (Odysseus format), mapped to the 14-combo set of the human data
    (mjg.actions.buttons_to_action rules); held with the Odysseus timing (30 frames with A, 10 with a
    direction, else 1, then one NOOP frame)
  - Glimmer's own thinking channel at --strength (low by default), the Odysseus tags in the answer

Saved per episode in <out>/<level>/run-<r>/ (as Odysseus): steps.jsonl (raw reply, thinking, tags, buttons,
combo, frames held, tokens, seconds), inputs.npy (every frame's input: replays the episode exactly),
ram.npy (RAM at each turn start), meta.json; hidden.npy with --save-hidden.

  python glimmer_play.py --levels w1l1 w8l3 --runs 4 --lora <SFT lora.pt> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "odysseus"))
from play import Episode, tag, VALID, OUR_LEVELS  # noqa: E402  (experiments/odysseus/play.py)

ROOT = os.environ.get("MJG_ROOT", ".")
MODELS = os.environ.get("MJG_MODELS", os.path.join(ROOT, "models"))
MODEL_DIR = f"{MODELS}/Muse-Glimmer-30B"
OUT = f"{ROOT}/results/glimmer_play/zeroshot_v1"

PROMPT = """You are playing Super Mario Bros. on the NES.

The goal is to progress through levels, collect coins and power-ups when safe, and ultimately finish the game by rescuing Princess Peach.

You can control the game by pressing buttons on the NES.

Available buttons:
- 'a': Jump (used to make Mario jump)
- 'b': Run/Shoot (hold to run faster or shoot fireballs if available)
- 'up': Climb ladders or vines (if present)
- 'down': Crouch or enter pipes (when standing on a pipe)
- 'left': Move Mario left
- 'right': Move Mario right
- 'noop': Do nothing (used to wait for a brief period without performing any action)

The game is paused while you think. Above are the frames of the last few seconds of play (oldest first); the last one is the current screen.

Please analyze the game screen and decide which buttons to press to progress.

Return your answer as follows:
1. Button sequence: a list of buttons to press simultaneously
2. Each button should be one of: 'a', 'b', 'up', 'down', 'left', 'right', 'noop'

First describe what you see on the screen in <perception></perception>. Then, in <reasoning></reasoning>, break down your reasoning step by step, justifying each action you consider. Output your final action in <answer>['button1', 'button2', ...]</answer>.

The maximum number of buttons you can press simultaneously in one turn is 3."""


def parse(reply, max_buttons=3):
    """Odysseus rule (experiments/odysseus/play.py parse), with up to 3 buttons."""
    import ast
    m = re.findall(r"<answer>(.*?)</answer>", reply, re.DOTALL)
    if not m:
        return None, [], "no <answer>"
    try:
        lst = ast.literal_eval(m[-1].strip())
        if not isinstance(lst, (list, tuple)):
            raise ValueError("not a list")
    except Exception as e:  # noqa: BLE001
        return None, [], f"unparsable answer: {e}"
    btn = []
    for b in lst:
        btn.append(str(b).lower())
    if len(btn) > max_buttons:
        return None, [], f"more than {max_buttons} buttons"
    valid, invalid = [], []
    for b in btn:
        if b in VALID:
            valid.append(b)
        else:
            invalid.append(b)
    return valid, invalid, None


def combo_buttons(combo):
    """14-combo index -> its button names, lowercased (empty for NOOP)."""
    from mjg.actions import COMBOS
    out = []
    for c in COMBOS[combo]:
        out.append(c.lower())
    return out


def answer_text(combo):
    """Canonical answer of a 14-combo, as kept in the history (same for the agent and the SFT data)."""
    b = combo_buttons(combo) or ["noop"]
    return f"<answer>{b}</answer>"


def to_combo(buttons):
    """Odysseus button names -> 14-combo index (human projection rules) and the NES buttons actually held."""
    from mjg.actions import ENV_BUTTONS, buttons_to_action
    name = {"a": "A", "b": "B", "up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT"}
    pressed = np.zeros((1, len(ENV_BUTTONS)), bool)
    for b in buttons:
        if b in name:
            pressed[0, ENV_BUTTONS.index(name[b])] = True
    a, _ = buttons_to_action(pressed)
    return int(a[0]), combo_buttons(int(a[0]))


def hold_frames(held):
    """Odysseus timing: 30 frames with A, 10 with a direction, else 1."""
    if "a" in held:
        return 30
    for d in ("left", "right", "up", "down"):
        if d in held:
            return 10
    return 1


class GEpisode(Episode):
    """Odysseus episode that also returns the frames seen during the hold.
    prefix = (bk2 path, n frames): instead of skipping the level intro, start from a human play's own start state and
    replay its first n frames (a scenario: the agent takes over at that moment). The prefix inputs are kept in
    inputs.npy and meta.json gets start_bk2, so replay.py reproduces the whole game."""
    prefix = None

    def skip_start(self, max_frames=600):
        if not self.prefix:
            return super().skip_start(max_frames)
        import retro
        bk2, n = self.prefix
        movie = retro.Movie(bk2)
        movie.step()
        self.env.initial_state = movie.get_state()
        self.obs, _ = self.env.reset()
        nb = self.env.num_buttons
        for _ in range(n):
            if not movie.step():
                break
            keys = []
            for i in range(nb):
                keys.append(movie.get_key(i, 0))
            self.obs, *_ = self.env.step(keys)
            self.inputs.append(keys[: len(self.names)])
        self.start_frames = len(self.inputs)

    def finish(self, meta):
        if self.prefix:
            meta = dict(meta, start_bk2=self.prefix[0], prefix_frames=self.start_frames)
        super().finish(meta)

    def step_frames(self, held, every):
        hold = hold_frames(held)
        frames, n = [], 0
        for i in range(hold):
            n += 1
            self.reason = self._frame(held)
            if i % every == every - 1:
                frames.append(self.obs.copy())
            if self.reason:
                break
        if hold > 1 and not self.reason:
            n += 1
            self.reason = self._frame([])
        frames.append(self.obs.copy())
        self.turns += 1
        if not self.reason and self.turns >= self.max_turns:
            self.reason = "max_turns"
        self.done = self.reason is not None
        self.rams.append(self.ram())
        return n, frames


def _worker(conn, integration, level, run, out_dir, max_turns, every, prefix=None):
    import site
    for sp in site.getsitepackages():
        if sp not in sys.path:
            sys.path.append(sp)
    GEpisode.prefix = prefix
    ep = GEpisode(integration, level, run, out_dir)
    ep.max_turns = max_turns
    conn.send(("ready", [ep.obs.copy()], ep.ram()))
    while True:
        cmd, arg = conn.recv()
        if cmd == "step":
            held, record = arg
            x_before = ep.mario_x()
            n, frames = ep.step_frames(held, every)
            record.update(turn=ep.turns, frames_held=n, x_before=x_before, x_after=ep.mario_x(),
                          progress=ep.mario_x() - ep.x0, done=ep.done, reason=ep.reason)
            ep.log.write(json.dumps(record) + "\n")
            ep.log.flush()
            conn.send((frames, ep.done, ep.reason, ep.turns, ep.mario_x() - ep.x0, ep.ram()))
        elif cmd == "finish":
            ep.finish(arg)
            conn.send("ok")
            return


class Remote:
    """A game in its own process. wait=False returns at once (start several games in parallel, then wait_ready())."""
    def __init__(self, ctx, a, level, run, wait=True, name=None, prefix=None):
        self.level, self.run, self.name = level, run, name or level  # name: output folder (a scenario's name)
        self.conn, child = ctx.Pipe()
        out_dir = os.path.join(a.out, self.name, f"run-{run:03d}")
        args = (child, a.integration, level, run, out_dir, a.max_turns, a.every) + ((prefix,) if prefix else ())
        self.proc = ctx.Process(target=_worker, args=args, daemon=True)
        # spawn re-runs the caller's script in every worker before it starts (for rl.py: torch, transformers, ...),
        # which can take minutes per game; the worker only needs this module, so hide the caller's path.
        # Not when this file is the script itself (then _worker lives in __main__ and the worker needs it).
        main = sys.modules["__main__"]
        main_file = main.__dict__.pop("__file__", None) if _worker.__module__ != "__main__" else None
        try:
            self.proc.start()
        finally:
            if main_file is not None:
                main.__file__ = main_file
        self.done, self.reason, self.turns, self.progress = False, None, 0, 0
        self.history = []  # [(frames, answer text)] of past turns
        if wait:
            self.wait_ready()

    def wait_ready(self):
        _, self.frames, self.ram = self.conn.recv()  # ram: RAM at the current turn start

    def step(self, held, record):
        self.conn.send(("step", (held, record)))
        self.frames, self.done, self.reason, self.turns, self.progress, self.ram = self.conn.recv()

    def finish(self, meta):
        self.conn.send(("finish", meta))
        self.conn.recv()
        self.proc.join(timeout=60)


def images(frames):
    """Frames at 1x, the last one (the current screen) at 2x."""
    from PIL import Image
    out = []
    for f in frames:
        out.append(Image.fromarray(f))
    last = out[-1]
    out[-1] = last.resize((last.width * 2, last.height * 2), Image.NEAREST)
    return out


def message_images(msgs):
    """The images of a conversation, in order (for the processor's images= argument)."""
    out = []
    for m in msgs:
        if not isinstance(m["content"], list):
            continue
        for c in m["content"]:
            if c.get("type") == "image":
                out.append(c["image"])
    return out


def chunk_vision(model, n=8):
    """Llama 3.2 Vision (Mllama) sends every image of a prompt through its vision encoder at once, each as 4 tile slots
    of 1,601 patches with a dense mask over them: >11 GB for one prompt with 8 past turns of screens. Run the images in
    chunks of n instead: the encoder treats each image on its own, so the output is the same. No-op for other models."""
    vm = getattr(getattr(model, "model", model), "vision_model", None)
    if type(vm).__name__ != "MllamaVisionModel":
        return
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    fwd = vm.forward

    def chunked(pixel_values=None, aspect_ratio_ids=None, aspect_ratio_mask=None, **kw):
        m = pixel_values.shape[1]
        if m <= n:
            return fwd(pixel_values=pixel_values, aspect_ratio_ids=aspect_ratio_ids,
                       aspect_ratio_mask=aspect_ratio_mask, **kw)
        parts = []
        for i in range(0, m, n):
            parts.append(fwd(pixel_values=pixel_values[:, i:i + n], aspect_ratio_ids=aspect_ratio_ids[:, i:i + n],
                             aspect_ratio_mask=aspect_ratio_mask[:, i:i + n], **kw).last_hidden_state)
        return BaseModelOutput(last_hidden_state=torch.cat(parts, 1))
    vm.forward = chunked


def build_conversation(past_frames, cur_frames, strength, player=None):
    """past_frames: frame lists of the previous turns (oldest first); cur_frames: frames since the last action.
    One user message with every frame (oldest first) then the prompt; no past answers (copycat problem).
    Shared by the agent (glimmer_play, rl) and the SFT data (train_sft), so both see the same format.
    player: optional demonstrator identity ("sub-03") for models trained on several people (identity
    conditioning, as in multi-demonstrator imitation, e.g. ILEED, Beliaev et al. 2022)."""
    frames = []
    for fr in past_frames:
        frames.extend(fr)
    frames.extend(cur_frames)
    content = []
    for im in images(frames):
        content.append({"type": "image", "image": im})
    content.append({"type": "text", "text": PROMPT})
    system = f"Reasoning strength: {strength}" + (f"\nPlay like player {player}." if player else "")
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def conversation(ep, k, strength, player=None):
    past = []
    for fr, _ in ep.history[-k:]:
        past.append(fr)
    return build_conversation(past, ep.frames, strength, player)


def split_thinking(raw):
    """Glimmer writes its thinking on the to=self channel, then the answer; returns (thinking, answer text)."""
    m = re.search(r"to=self<\|message\|>(.*?)<\|eom\|>", raw, re.S)
    thinking = m.group(1).strip() if m else None
    ans = raw[m.end():] if m else raw
    ans = re.sub(r"<\|[a-z_]+\|>", " ", ans)
    ans = re.sub(r"^\s*assistant\s*", "", ans).strip()
    return thinking, ans


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", nargs="+", default=OUR_LEVELS)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--k", type=int, default=8, help="past turns kept in the conversation")
    ap.add_argument("--every", type=int, default=10, help="keep every n-th frame of a hold (plus the last)")
    ap.add_argument("--strength", default="low", choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--integration", default=os.environ.get("MJG_INTEGRATION"))
    ap.add_argument("--worker-python", default=os.environ.get("ODY_WORKER_PY"))
    ap.add_argument("--lora", default=None, help="LoRA adapter (.pt from train_sft.py / rl.py) to load on Glimmer")
    ap.add_argument("--player", default=None, help="player to imitate, for models trained on several subjects")
    ap.add_argument("--save-hidden", action="store_true",
                    help="save the agent's internal state per turn: hidden state of the last prompt token at 4 depths "
                         "(decoder layers 13/26/39/52) -> <run dir>/hidden.npy [n_turns, 4, hidden] float16")
    ap.add_argument("--prefill", default="",
                    help="text the reply is forced to start with, e.g. '<perception>' (\\n = newline)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="chat template with enable_thinking=False (Qwen3.5); other models (Odysseus, Llama) ignore it")
    ap.add_argument("--scenarios", default=None,
                    help="json list of {name, level, bk2, frame}: each game starts from that human play at that frame "
                         "(replayed), then the agent plays (brain_pred/danger/scenarios.py); --levels / --runs ignored")
    a = ap.parse_args()
    a.prefill = a.prefill.replace("\\n", "\n")
    return a


def load_model(a):
    """Processor and model (bf16, one GPU if it has >= 80 GB, else split over the GPUs), with the LoRA if given."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    has_tokenizer = os.path.exists(f"{a.model_dir}/tokenizer_config.json")
    proc = AutoProcessor.from_pretrained(a.model_dir if has_tokenizer else MODEL_DIR)
    proc.tokenizer.padding_side = "left"
    if proc.tokenizer.pad_token_id is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    if torch.cuda.get_device_properties(0).total_memory < 80e9:
        mem = {}
        for g in range(torch.cuda.device_count()):
            mem[g] = "36GiB" if g == 0 else "44GiB"
        kw = dict(device_map="auto", max_memory=mem)
    else:
        kw = dict(device_map={"": 0})
    model = AutoModelForImageTextToText.from_pretrained(a.model_dir, dtype=torch.bfloat16, **kw).eval()
    chunk_vision(model)
    if a.lora:
        from mjg.lora import add_lora, find_layers
        _, layers = find_layers(model, model.config.text_config.num_hidden_layers)
        add_lora(layers, 16)
        sd = torch.load(a.lora)
        params = dict(model.named_parameters())
        with torch.no_grad():
            for nm, v in sd.items():
                params[nm].copy_(v.to(params[nm].device))
        model.eval()
        print(f"loaded LoRA {a.lora} ({len(sd)} tensors)", flush=True)
    return proc, model


def hidden_hooks(model, cap):
    """Forward hooks that keep, in cap[layer], the first call of each generate (the prompt), last position only.
    -> the hooked decoder layer indices (4 depths)."""
    from mjg.lora import find_layers
    _, layers = find_layers(model, model.config.text_config.num_hidden_layers)
    hid_layers = []
    for q in (1, 2, 3, 4):
        hid_layers.append(len(layers) * q // 4 - 1)

    def hook(i):
        def f(mod, inp_, out):
            o = out[0] if isinstance(out, tuple) else out
            if i not in cap and o.shape[1] > 1:
                cap[i] = o[:, -1].detach().float().cpu()
        return f
    for i in hid_layers:
        layers[i].register_forward_hook(hook(i))
    return hid_layers


def games_to_play(a):
    """[(level, run, Remote kwargs)] of the games without a meta.json yet."""
    todo = []
    if a.scenarios:
        for sc in json.load(open(a.scenarios)):
            if not os.path.exists(os.path.join(a.out, sc["name"], "run-000", "meta.json")):
                todo.append((sc["level"], 0, dict(name=sc["name"], prefix=(sc["bk2"], int(sc["frame"])))))
        return todo
    for lv in a.levels:
        for r in range(a.runs):
            if not os.path.exists(os.path.join(a.out, lv, f"run-{r:03d}", "meta.json")):
                todo.append((lv, r, {}))
    return todo


def main():
    a = parse_args()
    import torch
    proc, model = load_model(a)
    chat_kw = dict(enable_thinking=False) if a.no_thinking else {}
    cap, hid_layers = {}, []
    if a.save_hidden:
        hid_layers = hidden_hooks(model, cap)
    torch.manual_seed(a.seed)
    settings = dict(model_dir=a.model_dir, lora=a.lora, player=a.player, prefill=a.prefill, k=a.k, every=a.every,
                    strength=a.strength, max_new_tokens=a.max_new, temperature=1.0, top_p=0.95, top_k=64,
                    max_turns=a.max_turns, prompt=PROMPT,
                    timing="odysseus holds: 30 frames with A, 10 with a direction, else 1; then one NOOP")

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    if a.worker_python:
        ctx.set_executable(a.worker_python)
    todo = games_to_play(a)
    live, t0, calls, toks = [], time.time(), 0, 0
    while live or todo:
        while todo and len(live) < a.batch:
            lv, r, kw = todo.pop(0)
            live.append(Remote(ctx, a, lv, r, **kw))
        msgs = []
        for ep in live:
            msgs.append(conversation(ep, a.k, a.strength, a.player))
        if a.prefill:  # the reply is forced to start with this text (kept as part of the recorded reply)
            texts, imgs = [], []
            for m in msgs:
                texts.append(proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **chat_kw)
                             + a.prefill)
                imgs.append(message_images(m))
            inp = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to(model.device)
        else:
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt", padding=True, **chat_kw).to(model.device)
        inp.pop("token_type_ids", None)
        t1 = time.time()
        cap.clear()
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=a.max_new, do_sample=True, temperature=1.0, top_p=0.95, top_k=64)
        if a.save_hidden:
            per_layer = []
            for li in hid_layers:
                per_layer.append(cap[li])
            hb = torch.stack(per_layer, 1).half().numpy()  # [batch, 4, hidden]
            for j, ep in enumerate(live):
                if not hasattr(ep, "hidden"):
                    ep.hidden = []
                ep.hidden.append(hb[j])
        dt = time.time() - t1
        new = gen[:, inp["input_ids"].shape[1]:]
        n_tok = (new != proc.tokenizer.pad_token_id).sum(1).tolist()
        calls += len(live)
        toks += sum(n_tok)
        for ep, ids, nt in zip(live, new, n_tok):
            raw = a.prefill + proc.tokenizer.decode(ids, skip_special_tokens=False)
            thinking, reply = split_thinking(raw)
            buttons, invalid, err = parse(reply)
            combo, held = to_combo(buttons or []) if not err else (0, [])
            record = dict(raw=raw, reply=reply, thinking=thinking, perception=tag(reply, "perception"),
                          reasoning=tag(reply, "reasoning"), answer=tag(reply, "answer"), buttons=buttons,
                          invalid_buttons=invalid, error=err, combo=combo, held=held, new_tokens=int(nt),
                          context_tokens=int(inp["input_ids"].shape[1]), gen_seconds=round(dt, 2))
            ep.history.append((ep.frames, answer_text(combo)))
            ep.step(held, record)
        running = []
        for ep in live:
            if not ep.done:
                running.append(ep)
                continue
            ep.finish(dict(settings))
            if a.save_hidden and getattr(ep, "hidden", None):
                np.save(os.path.join(a.out, ep.name, f"run-{ep.run:03d}", "hidden.npy"), np.stack(ep.hidden))
            print(f"{ep.level} run {ep.run}: {ep.turns} turns, progress {ep.progress}, {ep.reason} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        live = running
        print(f"turn batch: {len(n_tok)} eps, {dt:.1f} s, mean new tokens {np.mean(n_tok):.0f}, "
              f"context {inp['input_ids'].shape[1]}; total {calls} calls, {toks / max(1, time.time() - t0):.1f} tok/s",
              flush=True)


if __name__ == "__main__":
    main()
