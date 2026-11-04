"""Odysseus plays Super Mario Bros (NES): their cross-game transfer setup, adjusted to our game and data.

Reproduces the Odysseus evaluation (github.com/odysseus-project/odysseus-mario-evaluation, smb.sh +
game_envs/super_mario_bros) and saves every reasoning trace:
  - model Odysseus-Project/Odysseus (Qwen3-VL-8B fine-tune), sampling temperature 1.0, top_p 1.0
  - one observation per turn, no history (--max_history 0), no text info; frame upscaled to 1280x1152
  - their prompt (env_prompts.SYSTEM_PROMPT = eval OLD_SYSTEM_PROMPT) with the game adjusted:
    Super Mario Land -> Super Mario Bros, Game Boy -> NES, Princess Daisy -> Princess Peach
  - action = up to 2 of a, b, up, down, left, right, noop; held 30 frames if it contains 'a', 10 if it
    contains a direction, else 1; one NOOP frame after a hold > 1 (as their env); up to 1000 turns
  - episode ends on death (player_state 0x06 / 0x0B or falling below the screen), level completed,
    time out or 1000 turns (as gym_super_mario_bros: is_dying / is_dead / flag_get)
Adjusted to our case: our emulator (stable-retro + the CNeuroMod mario.stimuli integration, the states
the participants played from), our 22 levels, transformers batched generation instead of a vLLM server.

Saved per episode in <out>/<level>/run-<r>/:
  steps.jsonl  one line per turn: full raw reply, <perception>, <reasoning>, parsed buttons, invalid
               buttons / parse error, frames held, Mario x, progress, done reason
  inputs.npy   bool [n_frames, n_buttons]: every emulator frame's input (replaying it reproduces the
               episode frame by frame, so screens are not stored)
  ram.npy      uint8 [n_turns + 1, 2048]: full RAM at each observation (turn start) and at the end
  meta.json    settings, prompt, outcome
Resumable: episodes with meta.json are skipped.

  python play.py --levels w1l1 --runs 2         # GPU; experiments/odysseus/job.sbatch runs every level
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "upstream"))
from mjg import emulator as E  # noqa: E402

ROOT = os.environ.get("MJG_ROOT", ".")
MODELS = os.environ.get("MJG_MODELS", os.path.join(ROOT, "models"))
MODEL_DIR = f"{MODELS}/Odysseus"
OUT = f"{ROOT}/results/odysseus_play/v1"
OUR_LEVELS = ["w1l1", "w1l2", "w1l3", "w2l1", "w2l3", "w3l1", "w3l2", "w3l3", "w4l1", "w4l2", "w4l3",
              "w5l1", "w5l2", "w5l3", "w6l1", "w6l2", "w6l3", "w7l1", "w7l3", "w8l1", "w8l2", "w8l3"]
VALID = ["a", "b", "up", "down", "left", "right", "noop"]
IMG_W, IMG_H = 1280, 1152
MAX_TURNS = 1000


def make_prompt():
    from env_prompts import SYSTEM_PROMPT
    p = SYSTEM_PROMPT
    for a, b in (("Super Mario Land", "Super Mario Bros."), ("Game Boy", "NES"),
                 ("Princess Daisy", "Princess Peach")):
        assert a in p, a
        p = p.replace(a, b)
    return p


def parse(reply):
    """Their rule: the <answer> list, lowercased, at most 2 buttons, unknown names reported as invalid."""
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
    if len(btn) > 2:
        return None, [], "more than 2 buttons"
    valid, invalid = [], []
    for b in btn:
        if b in VALID:
            valid.append(b)
        else:
            invalid.append(b)
    return valid, invalid, None


def tag(reply, name):
    m = re.findall(rf"<{name}>(.*?)</{name}>", reply, re.DOTALL)
    return m[-1].strip() if m else None


class Episode:
    """One Odysseus episode on our emulator, with their action timing."""

    def __init__(self, integration, level, run, out_dir):
        self.level, self.run, self.dir = level, run, out_dir
        w, l = level[1], level[3]
        self.env = E.make_env(integration, state=f"Level{w}-{l}")
        obs, _ = self.env.reset()
        self.names = []
        for b in self.env.buttons:
            self.names.append(str(b))
        self.idx = {}
        for i, n in enumerate(self.names):
            if n != "None":
                self.idx[n.lower()] = i
        self.obs, self.inputs, self.turns = obs, [], 0
        self.skip_start()
        self.rams = [self.ram()]
        self.x0 = self.mario_x()
        self.lives0, self.done, self.reason = int(self.ram()[0x75A]), False, None
        self.level_ids = (int(self.ram()[0x75F]), int(self.ram()[0x75C]))
        os.makedirs(out_dir, exist_ok=True)
        self.log = open(os.path.join(out_dir, "steps.jsonl"), "w")

    def timer(self):
        r = self.ram()
        return int(r[0x7F8]) * 100 + int(r[0x7F9]) * 10 + int(r[0x7FA])

    def skip_start(self, max_frames=600):
        """Like gym_super_mario_bros: no-op through the level intro until Mario is in control (normal state,
        game timer running). These frames are kept in inputs.npy so the episode replays exactly."""
        t0 = self.timer()
        for _ in range(max_frames):
            if self.ram()[0x0E] == 8 and self.timer() < t0:
                break
            a = [0] * self.env.num_buttons
            self.obs, *_ = self.env.step(a)
            self.inputs.append(a[: len(self.names)])
        self.start_frames = len(self.inputs)

    def ram(self):
        return np.array(self.env.get_ram(), dtype=np.uint8)

    def mario_x(self):
        r = self.ram()
        return int(r[0x6D]) * 256 + int(r[0x86])

    def _frame(self, buttons):
        a = [0] * self.env.num_buttons
        for b in buttons:
            if b in self.idx:
                a[self.idx[b]] = 1
        self.obs, _, term, trunc, _ = self.env.step(a)
        self.inputs.append(a[: len(self.names)])
        r = self.ram()
        if r[0x0E] in (0x06, 0x0B) or r[0xB5] > 1 or int(r[0x75A]) < self.lives0:
            return "death"
        if (int(r[0x75F]), int(r[0x75C])) != self.level_ids or r[0x1D] == 3:
            return "level_completed"
        t = int(r[0x7F8]) * 100 + int(r[0x7F9]) * 10 + int(r[0x7FA])
        if t == 0 and self.turns > 0:
            return "time_out"
        return "ended" if (term or trunc) else None

    def step(self, buttons):
        """Their timing: 30 frames with 'a', 10 with a direction, else 1; then one NOOP frame."""
        nes = []
        for b in buttons:
            if b != "noop":
                nes.append({"a": "a", "b": "b", "up": "up", "down": "down", "left": "left", "right": "right"}[b])
        hold = 1
        for d in ("left", "right", "up", "down"):
            if d in nes:
                hold = 10
                break
        if "a" in nes:
            hold = 30
        n = 0
        for _ in range(hold):
            n += 1
            self.reason = self._frame(nes)
            if self.reason:
                break
        if hold > 1 and not self.reason:
            n += 1
            self.reason = self._frame([])
        self.turns += 1
        if not self.reason and self.turns >= MAX_TURNS:
            self.reason = "max_turns"
        self.done = self.reason is not None
        self.rams.append(self.ram())
        return n

    def skip(self):
        """Unparsable reply: the game does not advance (their loop returns the same observation)."""
        self.turns += 1
        if self.turns >= MAX_TURNS:
            self.reason, self.done = "max_turns", True
        self.rams.append(self.ram())
        return 0

    def finish(self, meta):
        self.log.close()
        np.save(os.path.join(self.dir, "inputs.npy"), np.asarray(self.inputs, dtype=bool))
        np.save(os.path.join(self.dir, "ram.npy"), np.stack(self.rams))
        meta.update(level=self.level, run=self.run, turns=self.turns, frames=len(self.inputs),
                    end=self.reason, x_start=self.x0, x_end=self.mario_x(),
                    progress=self.mario_x() - self.x0, button_names=self.names, start_frames=self.start_frames)
        json.dump(meta, open(os.path.join(self.dir, "meta.json"), "w"), indent=1)
        self.env.close()


def _worker(conn, integration, level, run, out_dir, max_turns):
    """Owns one Episode in its own process (stable-retro allows one emulator per process)."""
    global MAX_TURNS
    MAX_TURNS = max_turns
    import site                               # spawn copies the parent's sys.path: add this interpreter's
    for sp in site.getsitepackages():         # own site-packages (stable-retro lives in the worker env)
        if sp not in sys.path:
            sys.path.append(sp)
    ep = Episode(integration, level, run, out_dir)
    conn.send(("ready", ep.obs, ep.x0))
    while True:
        cmd, arg = conn.recv()
        if cmd == "step":
            buttons, record = arg
            x_before = ep.mario_x()
            held = ep.step(buttons) if buttons is not None else ep.skip()
            record.update(turn=ep.turns, frames_held=held, x_before=x_before, x_after=ep.mario_x(),
                          progress=ep.mario_x() - ep.x0, done=ep.done, reason=ep.reason)
            ep.log.write(json.dumps(record) + "\n")
            ep.log.flush()
            conn.send((ep.obs, ep.done, ep.reason, ep.turns, ep.mario_x() - ep.x0))
        elif cmd == "finish":
            ep.finish(arg)
            conn.send("ok")
            return


class Remote:
    """Main-process handle of an Episode living in a worker process."""

    def __init__(self, ctx, integration, level, run, out_dir):
        self.level, self.run = level, run
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(child, integration, level, run, out_dir, MAX_TURNS),
                                daemon=True)
        self.proc.start()
        _, self.obs, _ = self.conn.recv()
        self.done, self.reason, self.turns, self.progress = False, None, 0, 0

    def image(self):
        from PIL import Image
        return Image.fromarray(self.obs).resize((IMG_W, IMG_H), Image.NEAREST)

    def step(self, buttons, record):
        self.conn.send(("step", (buttons, record)))
        self.obs, self.done, self.reason, self.turns, self.progress = self.conn.recv()

    def finish(self, meta):
        self.conn.send(("finish", meta))
        self.conn.recv()
        self.proc.join(timeout=60)


def episodes_to_play(a):
    """This shard's (level, run) pairs (every nshards-th of all of them, from shard) without a meta.json yet."""
    every = []
    for lv in a.levels:
        for r in range(a.runs):
            every.append((lv, r))
    todo = []
    for lv, r in every[a.shard::a.nshards]:
        if not os.path.exists(os.path.join(a.out, lv, f"run-{r:03d}", "meta.json")):
            todo.append((lv, r))
    return todo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", nargs="+", default=OUR_LEVELS)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16, help="episodes played in parallel (one generate per turn)")
    ap.add_argument("--max-new", type=int, default=8192, help="their max_tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--integration", default=os.environ.get("MJG_INTEGRATION"))
    ap.add_argument("--max-turns", type=int, default=1000, help="their --steps (1000)")
    ap.add_argument("--worker-python", default=os.environ.get("ODY_WORKER_PY"),
                    help="python for the emulator worker processes (needs stable-retro; same Python version)")
    a = ap.parse_args()
    global MAX_TURNS
    MAX_TURNS = a.max_turns

    prompt = make_prompt()
    todo = episodes_to_play(a)
    settings = dict(model_dir=a.model_dir, temperature=1.0, top_p=1.0, max_new_tokens=a.max_new, seed=a.seed,
                    image_size=[IMG_W, IMG_H], history=0, max_turns=a.max_turns, prompt=prompt,
                    source="odysseus-mario-evaluation smb.sh, adjusted (see play.py docstring)")
    print(f"{len(todo)} episodes to play", flush=True)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(a.model_dir)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(a.model_dir, dtype=torch.bfloat16).to("cuda")
    model.eval()
    torch.manual_seed(a.seed)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    if a.worker_python:                      # model env without stable-retro: emulators run in another env
        ctx.set_executable(a.worker_python)
    live, queue, t0 = [], list(todo), time.time()
    while live or queue:
        while queue and len(live) < a.batch:
            lv, r = queue.pop(0)
            live.append(Remote(ctx, a.integration, lv, r, os.path.join(a.out, lv, f"run-{r:03d}")))
        msgs = []
        for ep in live:
            msgs.append([{"role": "user", "content": [{"type": "image", "image": ep.image()},
                                                      {"type": "text", "text": prompt}]}])
        inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=a.max_new, do_sample=True, temperature=1.0, top_p=1.0,
                                 top_k=0)
        new = gen[:, inputs["input_ids"].shape[1]:]
        replies = processor.batch_decode(new, skip_special_tokens=True)
        n_tok = (new != processor.tokenizer.pad_token_id).sum(1).tolist()
        for ep, reply, nt in zip(live, replies, n_tok):
            buttons, invalid, err = parse(reply)
            record = dict(reply=reply, perception=tag(reply, "perception"), reasoning=tag(reply, "reasoning"),
                          answer=tag(reply, "answer"), buttons=buttons, invalid_buttons=invalid, error=err,
                          new_tokens=int(nt))
            ep.step(None if err else (buttons or ["noop"]), record)
        running = []
        for ep in live:
            if not ep.done:
                running.append(ep)
                continue
            ep.finish(dict(settings))
            print(f"{ep.level} run {ep.run}: {ep.turns} turns, progress {ep.progress}, {ep.reason} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        live = running


if __name__ == "__main__":
    main()
