# brain_pred: from Mario gameplay to brain activity

**An agent that generates brain activity.** The agent (Glimmer: VLA-JEPA perception, SFT on human play with its own
reasoning, RL toward one person) plays Mario; its internal state, mapped through an encoder fitted on human fMRI,
gives the brain activity its "mind" implies, scan by scan. That means something only if the agent is human-like and
brain-aligned, so the measurable goals are:
1. **Brain alignment:** the agent's state predicts held-out human BOLD (held-out fMRI sessions) beyond what is on
   screen (V-JEPA 2.1, a pure vision model) and the game state.
2. **Human-likeness:** the agent plays like the person and reasons on every move
   (`experiments/glimmer_imitation`).
3. **Reusable outputs:** the per-subject encoders (with their held-out accuracy) and the generator + demo (the agent
   plays -> its reasoning + internal state + generated BOLD).

## Method

**Calibration.** The agent watches the exact screens each participant saw in the scanner, cut into its own turns
(the same cutting as its SFT data), and at every decision gets the conversation it would have if it were playing
there: the frames of its last 8 turns, the current turn's frames and its prompt, nothing about the human's action.
Each scan gets the agent's state at its most recent decision. Two signals come out:
- **the agent's multimodal prompt state**: the hidden state of the last prompt token (images + text) at 4 depths of
  the language model (layers 13/26/39/52), kept only on scans whose decision is at most one scan old (in practice
  Mario's first life, `fresh_mask.py`);
- **the running context of its replies**: the agent writes its reply (perception, reasoning, answer) at every
  decision, then its own language model reads, text only and in order, every reply it wrote so far in the play; the
  state at the end of each reply carries it in the context of the earlier ones (causal, so only the past).

**Encoding.** Ridge regression from per-scan features to 1,000 Schaefer parcels, with FIR delays of 1-5 scans,
fitted on the training sessions and tested on the held-out sessions (`configs/holdout_sessions.json`). Every encoder
has a nuisance band (buttons, level, time in play, and with `--lowlevel` the screen luminance, optical flow and
audio envelope), so a feature only gets credit beyond those. Wide bands are reduced to 256 principal components
fitted on training scans only; groups joined by `&` get one ridge penalty each.

**Final encoder:** V-JEPA 2.1 (4 blocks) & the agent's prompt state (layer 26) & the running context of its replies
(layer 26), one penalty per group: `vjepa4&agent_sftfresh_d1&agent_sftctx_d1`. Its references are V-JEPA alone and
V-JEPA with each agent signal; the control shifts the reply context by half a run (timing destroyed). Results:
[top-level README](../README.md#results).

## Layout

| Folder | Step | What it does | Main entry point |
|---|---|---|---|
| `agent/` | **0. Calibration** (the agent on the scanner screens) | The agent's turns on every fMRI play, its prompt state per scan (4 depths), its written reply at each decision, the fresh-state mask, and the running context of its replies. `trace_features.py`: interpretable bands of what the replies talk about. | `build_scan_turns.py` (CPU) -> `embed_scans.py` (GPU) -> `fresh_mask.py`, `context_features.py` |
| `encoding/` | **1. Encoding** (fit + evaluate) | Per-scan features of each human run -> ridge with FIR delays -> tested on the held-out sessions. `--save` writes each fitted encoder for reuse. `region_report.py`: where in the brain (7 networks) each encoder helps. | `encode.py` (+ `build_tr_features.py`) |
| `features/` | shared | Per-scan features of any play (agent or person) with the training definitions (buttons, game state from RAM, level, time), and the low-level confounds (luminance, optical flow, audio). | `play_features.py`, `lowlevel.py` |
| `prediction/` | **2. Prediction** (apply) | Loads a saved encoder and turns per-scan features into predicted BOLD [n_scans, 1000], with the training preprocessing. | `predict.py` (`EncoderModel`) |
| `generator/` | **3. Generator** (end to end) | A recorded play -> replay (frames + RAM) -> features -> prediction, for one or more subjects. | `generate_bold.py` (+ `replay.py`) |
| `demo/` | demo | Video: the agent playing, its reasoning, its internal state and the predicted brain, in sync. | `make_demo.sh` (`render.py`) |
| `danger/` | analysis | How the human brain and the agent respond to danger (risk of dying within 3 s, from RAM), the agent's danger direction, and steering what the agent writes along it. | `human_response.py`, `danger_direction.py`, `steer_model.sh` |
| `LITERATURE.md` | | Related work. | |

Data flow:

```
human fMRI runs --(encoding/build_tr_features.py)--> game + V-JEPA features per scan ----------------+
human fMRI runs --(agent/build_scan_turns, embed_scans, fresh_mask)--> AGENT'S STATE per scan -------+--(encoding/encode.py --save)--> encoder per subject
                --(agent/embed_scans --reason, context_features)--> CONTEXT OF ITS REPLIES per scan --+                                       |
agent plays (glimmer_play --save-hidden) --(generator/replay.py)--> frames + RAM + its state per turn --(features/)--> per scan --(prediction/)--> generated BOLD
```
The agent's state is defined the same way in calibration and in its own play. The predicted BOLD is a model
prediction, not a measurement; how good it is = the encoder's held-out accuracy, stored with it.

## How to run

Paths come from environment variables (`MJG_ROOT`, `MJG_SCRATCH`, `MJG_CNEUROMOD`, ...): copy
`slurm/paths.example.sh` to `slurm/paths.local.sh` and fill it in; every launcher sources `slurm/env.sh`. Features
and results go under `$MJG_ROOT/results/brain_encoding/`, the agent's turns under `$MJG_SCRATCH/glimmer_scan/`.

```bash
# 1. per-scan features of the human runs, and the low-level confounds for the nuisance band (CPU)
python brain_pred/encoding/build_tr_features.py --subjects sub-03
python brain_pred/features/lowlevel.py --subjects sub-03 --procs 32
python brain_pred/encoding/build_tr_features.py --subjects sub-03 --extra lowlevel
# 2. calibration: the agent's turns on the scanner screens (CPU); its prompt state and its reply at every decision
#    (4 GPUs); the fresh-state mask; the running context of its replies (GPU)
sbatch brain_pred/agent/job_scan_turns.sbatch sub-03
python brain_pred/agent/embed_scans.py --subject sub-03 --tag sft --lora <SFT lora.pt> --model-dir <VLA-JEPA Glimmer> --player sub-03 --reason
python brain_pred/agent/fresh_mask.py --subject sub-03 --src sft
python brain_pred/agent/context_features.py --subject sub-03 --tag sft --lora <SFT lora.pt> --model-dir <VLA-JEPA Glimmer>
# 3. the final encoder, its references and the timing control, saved for reuse; then where it helps
M="nuis,vjepa4,agent_sftfresh_d1,vjepa4&agent_sftfresh_d1,vjepa4&agent_sftctx_d1,vjepa4&agent_sftfresh_d1&agent_sftctx_d1,vjepa4&agent_sftctx_d1_shift"
python brain_pred/encoding/encode.py --subjects sub-03 --lowlevel --agent-tags sftfresh sftctx --models "$M" --save $MJG_ROOT/results/bold_generator
python brain_pred/encoding/region_report.py --subject sub-03 --suffix _lowlevel --models "$M" --compare "vjepa4&agent_sftctx_d1,vjepa4" --margin 0.005
# 4. BOLD generated from the agent's own play (recorded with glimmer_play.py --save-hidden), and the demo video
python brain_pred/generator/generate_bold.py --play <play dir> --subjects sub-03 --model <encoder name> --out bold.npz
bash brain_pred/demo/make_demo.sh <play dir> demo.mp4 sub-03 <encoder name>
```

On a Slurm cluster, `agent/job_embed.sbatch` runs `embed_scans.py` as an array over runs, and `agent/traces_lm.sh`
runs its two passes (the replies, then the prompt states) as pooled workers on the all-lives turns
(`job_scan_turns.sbatch sub-03 --all-lives`; then pass `--turns-split all_lives` to `fresh_mask.py` and
`context_features.py`). `encoding/job.sbatch` runs `build_tr_features.py` (without the low-level band) and `encode.py` with its default
models.
