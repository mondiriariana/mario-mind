# Mario Mind

*An AI that plays like a person, thinks out loud, and predicts that person's brain activity.*

![The agent sees the screen a person saw in the scanner, reasons about it, and its internal state predicts that person's brain activity](docs/figures/hero.svg)

**Goal: an AI agent whose "mind" predicts a person's brain.** Muse Glimmer 30B learns to play *Super Mario Bros.*
the way a specific person does and says what it sees and why on every move. Its internal state, read through an
encoder fitted on that person's fMRI, predicts their brain activity scan by scan, on sessions it never saw. An agent
that plays like a person and predicts their brain is a **virtual participant**: hypotheses can be tested on its
generated brain data before paying for scanner time.

Built for the Meta Global AI Developer Hackathon (Oct 26 - Nov 4, 2026), theme *Push the Frontier*. Author: Ariana
Mondiri (Boston College). **Demo:** [the agent playing, its reasoning and the predicted brain](https://mondiriariana.github.io/arianamondiri/_site/project/2026-10-30-mario-agent/)
(the brain panel is a model prediction, not a measurement).

## How it works

![Pipeline: Glimmer 30B, VLA-JEPA, SFT, RL, brain](docs/figures/pipeline.svg)

1. **VLA-JEPA** (`experiments/glimmer_vlajepa`): Glimmer is fully fine-tuned to predict what happens next on screen,
   in the space of V-JEPA 2.1 (Meta's video model), 0.5 to 2 s ahead, with a language loss so it keeps reasoning.
2. **SFT** (`experiments/glimmer_imitation`): on 5 people's play, Glimmer first writes its own perception and
   reasoning for each moment (from the screens only), then learns to press the button the person pressed.
3. **RL** (`experiments/glimmer_imitation/rl.py`): the agent plays on its own, rewarded for playing like one person
   (GAIL), with a KL leash to the SFT model.
4. **Calibration** (`brain_pred/agent`): the agent watches the exact screens the person saw in the scanner; we record
   its internal state and the running context of what it has written.
5. **Brain encoder** (`brain_pred/encoding`): ridge regression from V-JEPA + the agent's state + its reply context to
   1,000 brain regions, fitted on training sessions and tested on held-out sessions. The generator and demo
   (`brain_pred/generator`, `brain_pred/demo`) apply it to the agent's own play.

Data: CNeuroMod's *Super Mario Bros.* dataset, 5 people with 20-30 fMRI sessions each, playing the real game in the
scanner with every button press recorded.

## Results

**The agent's mind adds to what is on screen** (sub-03, held-out sessions, gain in mean correlation over a nuisance
model of buttons, level, time, luminance, motion and sound; 1,000 regions):

| Encoder | Gain |
|---|---|
| V-JEPA 2.1 alone (the strongest single feature) | +0.0686 |
| **V-JEPA + the agent's state + the context of its replies** | **+0.0760** |
| Control: the reply context shifted in time | +0.0679 |

Adding the context of what the agent has written beats V-JEPA alone in 482 of 1,000 regions (somatomotor, salience,
control, attention and default-mode networks). Shifted in time it adds nothing, so the gain is tied to the moment.

**It plays like the person and reasons on every move** (SFT recipe, tested on one person, sub-01; 30 games each
against 9,432 held-out moves of that person; action KL = distance between the agent's and the person's button mix):

| Agent | Action KL vs the person | Reasons on every move |
|---|---|---|
| Glimmer out of the box | 3.31 | 95% |
| **Glimmer after SFT (own reasoning, then the person's button)** | **0.09** | **100%** |

Limits: brain results on one person and one held-out split so far; RL and the final agent are single runs. Every
metric is explained in [docs/METRICS.md](docs/METRICS.md).

## Meta models used

| Model | How |
|---|---|
| **Muse Glimmer 30B** (open weights) | Full fine-tune (VLA-JEPA, FSDP on 4 H200), LoRA SFT with its own reasoning, RL with a GAIL reward, served with vLLM for rollouts; its hidden states are the agent's "mind". All on our own hardware. |
| **V-JEPA 2.1 ViT-G** | Video features: the targets VLA-JEPA learns to predict, and a brain-encoding feature. |

## Repository

| Path | What |
|---|---|
| `experiments/glimmer_vlajepa/` | VLA-JEPA on Glimmer |
| `experiments/glimmer_imitation/` | play loop, SFT, RL, test against the person |
| `experiments/odysseus/` | Odysseus reasoning traces turned into SFT examples |
| `experiments/vjepa_spark_combo/` | the frame set VLA-JEPA trains on |
| `brain_pred/` | calibration, brain encoders, generator, demo, danger analysis ([README](brain_pred/README.md)) |
| `scripts/` | V-JEPA features, full game RAM, moment index ([README](scripts/README.md)) |
| `mjg/` | shared code: emulator, data index, V-JEPA loader, encoding helpers, LoRA |
| `slurm/`, `configs/` | cluster launchers, held-out sessions |
| `docs/` | data, metrics, figures |

## Reproduce

**Data (not in this repository):** CNeuroMod *mario* and *mario.fmriprep*, under CNeuroMod's data-use terms
(https://docs.cneuromod.ca); the game integration from
[courtois-neuromod/mario.stimuli](https://github.com/courtois-neuromod/mario.stimuli). Model weights: Muse Glimmer
30B, V-JEPA 2.1 ViT-G. **Compute:** 4x H200 (VLA-JEPA), 2-4x L40S (SFT, RL, calibration), CPU nodes (features,
encoders); paths in `slurm/paths.local.sh` (copy `slurm/paths.example.sh`). Held-out fMRI sessions are fixed in
`configs/holdout_sessions.json` and never used for training.

```bash
sbatch experiments/glimmer_vlajepa/job_long.sbatch                                # 1. VLA-JEPA (settings: long_run.sh)
sbatch --export=ALL,SFT_MODEL=<VLA-JEPA Glimmer> experiments/glimmer_imitation/job_sft.sbatch   # 2. SFT (sft.sh)
sbatch --export=ALL,RL_MODEL=<VLA-JEPA Glimmer>,RL_INIT=<SFT lora.pt>,RL_FORCE=1 \
       experiments/glimmer_imitation/job_rl.sbatch                                # 3. RL (settings: rl.sh)
# 4-5. calibration, brain encoder, generator, demo: brain_pred/README.md
```

## Built during the hackathon

Everything Glimmer (VLA-JEPA, SFT, RL, the play loop), the calibration, the brain encoders with the agent and its
reply context, the generator, the demo and the danger analysis were built during the hackathon (Oct 26 - Nov 4).
Reused from before: the emulator wrapper and data index (`mjg/emulator.py`, `mjg/data.py`, `mjg/objects.py`, Oct
21-24) and ridge / timing helpers from the author's earlier fMRI work (`mjg/encoding.py`).

## Credits

CNeuroMod (Courtois NeuroMod project) for the *mario* dataset and game integration; VLA-JEPA (arXiv 2602.10098) for
the world-model objective; Odysseus for the reasoning-trace format; Schaefer et al. 2018 for the brain atlas.
Related work: [brain_pred/LITERATURE.md](brain_pred/LITERATURE.md).
