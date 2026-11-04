"""vLLM generation for Glimmer rollouts (RL, play, reasoning traces), in its own process.

vLLM lives in its own Python environment ($MJG_VLLM_PY, with its own torch), so the training process ($MJG_MUSE_PY)
cannot import it. The trainer starts this server on the generation GPUs and talks to it over a local socket
(multiprocessing.connection, pickle):
  set_lora(path to our LoRA .pt)  -> converted to a PEFT adapter and registered under a new id (after every update)
  generate([(prompt text, [PIL images])], sampling) -> [(generated token ids, text, ...)], seconds
The prompt text is the chat template applied by the trainer (same function as the HF path); vLLM expands the image
placeholders itself. Our LoRA (mjg.lora: <module>.A [r, in], <module>.B [out, r], scale alpha / r with alpha = 2r)
maps onto PEFT names; vLLM packs q/k/v and gate/up and renames the attention gate (hf_to_vllm_mapper).

Inputs: $MJG_VLLM_PY (the vLLM Python), $MJG_MODELS/Muse-Glimmer-30B (default model).
  server (normally started by Client): CUDA_VISIBLE_DEVICES=0,1 $MJG_VLLM_PY vllm_engine.py --serve --tp 2 --port 6001
"""
import argparse, json, os, subprocess, time

ROOT = os.environ.get("MJG_ROOT", ".")
SCRATCH = os.environ.get("MJG_SCRATCH", os.path.join(ROOT, "scratch"))
MODELS = os.environ.get("MJG_MODELS", os.path.join(ROOT, "models"))
VLLM_PY = os.environ.get("MJG_VLLM_PY", "python")
# the vLLM venv is built on the emulator env's conda Python ($MJG_PY), whose lib/ holds the newer libstdc++ it needs
_EMU_PY = os.environ.get("MJG_PY")
BASE_LIB = os.path.join(os.path.dirname(os.path.dirname(_EMU_PY)), "lib") if _EMU_PY else None
MODEL_DIR = f"{MODELS}/Muse-Glimmer-30B"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
KEY = b"glimmer-vllm"


def lora_to_peft(sd, out_dir, base):
    """Write our LoRA state dict as a PEFT adapter folder (adapter_model.safetensors + adapter_config.json)."""
    import torch
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    t = {}
    r = None
    for k, v in sd.items():
        module, part = k.rsplit(".", 1)
        t[f"base_model.model.{module}.lora_{part}.weight"] = v.to(torch.bfloat16).contiguous()
        if r is None and k.endswith(".A"):
            r = v.shape[0]
    save_file(t, f"{out_dir}/adapter_model.safetensors")
    json.dump(dict(peft_type="LORA", r=r, lora_alpha=2 * r, target_modules=TARGETS, lora_dropout=0.0, bias="none",
                   base_model_name_or_path=base, task_type="CAUSAL_LM"), open(f"{out_dir}/adapter_config.json", "w"))


def _generate(llm, prompts, sp, lora):
    """Run one batch: -> ([(token ids, text, first 8 prompt ids, prompt length)], seconds)."""
    from vllm import SamplingParams
    inputs = []
    for text, imgs in prompts:
        inputs.append({"prompt": text, "multi_modal_data": {"image": imgs}})
    t = time.time()
    outs = llm.generate(inputs, SamplingParams(skip_special_tokens=False, **sp), lora_request=lora, use_tqdm=False)
    res = []
    for o in outs:
        res.append((list(o.outputs[0].token_ids), o.outputs[0].text, list(o.prompt_token_ids[:8]),
                    len(o.prompt_token_ids)))
    return res, time.time() - t


def serve(a):
    import torch
    from multiprocessing.connection import Listener
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16", enable_lora=True,
              max_lora_rank=a.max_lora_rank, max_loras=1, max_model_len=a.max_len, gpu_memory_utilization=a.mem, seed=0,
              limit_mm_per_prompt={"image": a.max_images}, enable_prefix_caching=True,
              disable_custom_all_reduce=a.no_custom_ar)  # L40S (PCIe, no P2P): the custom all-reduce fails
    lora, n = None, 0
    with Listener(("127.0.0.1", a.port), authkey=KEY) as lis:
        print("VLLM_READY", flush=True)
        conn = lis.accept()
        while True:
            cmd, arg = conn.recv()
            try:
                if cmd == "lora":
                    n += 1
                    d = f"{a.work}/lora_{n:05d}"
                    lora_to_peft(torch.load(arg, map_location="cpu"), d, a.model)
                    lora = LoRARequest(f"policy{n}", n, d)
                    prev = f"{a.work}/lora_{n - 2:05d}"  # keep the last two adapters on disk
                    if os.path.isdir(prev):
                        subprocess.run(["rm", "-rf", prev])
                    conn.send(("ok", n))
                elif cmd == "nolora":
                    lora = None
                    conn.send(("ok", 0))
                elif cmd == "gen":
                    prompts, sp = arg
                    conn.send(("ok", _generate(llm, prompts, sp, lora)))
                elif cmd == "stop":
                    conn.send(("ok", None))
                    break
            except Exception:  # noqa: BLE001  (report to the trainer instead of dying silently)
                import traceback
                conn.send(("error", traceback.format_exc()))


def _server_env(gpus):
    """Environment of the server process: its GPUs, runtime LoRA updates, the libraries and PATH of $MJG_VLLM_PY."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, VLLM_ALLOW_RUNTIME_LORA_UPDATING="1",
               # FlashInfer's sampler compiles kernels at first use (needs nvcc + ninja): use vLLM's PyTorch sampler
               VLLM_USE_FLASHINFER_SAMPLER="0")
    if BASE_LIB:
        old = os.environ.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = BASE_LIB + (":" + old if old else "")
    if os.path.dirname(VLLM_PY):
        env["PATH"] = os.path.dirname(VLLM_PY) + ":" + os.environ.get("PATH", "")
    return env


class Client:
    """Starts the server on `gpus` (a CUDA_VISIBLE_DEVICES string) and connects. Call before the trainer touches CUDA
    if the trainer itself restricts CUDA_VISIBLE_DEVICES afterwards."""
    def __init__(self, gpus, tp, work, port=6001, model=MODEL_DIR, mem=0.88, max_len=8192, max_images=64, log=None):
        os.makedirs(work, exist_ok=True)
        cmd = [VLLM_PY, "-u", os.path.abspath(__file__), "--serve", "--tp", str(tp), "--port", str(port),
               "--work", work, "--model", model, "--mem", str(mem), "--max-len", str(max_len),
               "--max-images", str(max_images)]
        self.log = open(log or f"{work}/server.log", "w")
        # own process group, so kill() also stops vLLM's GPU workers; atexit covers crashes of the trainer
        self.proc = subprocess.Popen(cmd, env=_server_env(gpus), stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.port, self.work, self.conn = port, work, None
        import atexit
        atexit.register(self.kill)

    def wait(self, timeout=1800):
        from multiprocessing.connection import Client as C
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM server died, see {self.log.name}")
            if "VLLM_READY" in open(self.log.name).read():
                self.conn = C(("127.0.0.1", self.port), authkey=KEY)
                return time.time() - t0
            time.sleep(5)
        raise TimeoutError("vLLM server not ready")

    def _call(self, cmd, arg=None):
        self.conn.send((cmd, arg))
        st, out = self.conn.recv()
        if st != "ok":
            raise RuntimeError(f"vLLM server: {out}")
        return out

    def set_lora(self, path):
        return self._call("lora", path)

    def no_lora(self):
        return self._call("nolora")

    def generate(self, prompts, **sp):
        return self._call("gen", (prompts, sp))

    def kill(self):
        import signal
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def close(self):
        try:
            self._call("stop")
            self.proc.wait(timeout=120)
        finally:
            self.kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--port", type=int, default=6001)
    ap.add_argument("--work", default=f"{SCRATCH}/tmp/vllm_lora")
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--mem", type=float, default=0.88)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--max-images", type=int, default=64)
    ap.add_argument("--max-lora-rank", type=int, default=16)
    ap.add_argument("--no-custom-ar", type=int, default=1,
                    help="1 = NCCL all-reduce (needed on L40S; 0 on NVLink GPUs)")
    a = ap.parse_args()
    if a.serve:
        serve(a)
