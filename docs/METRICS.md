# Metrics: what each number means and how to read it

All scores are on data the model was not trained on (validation or held-out test sessions). "Range / better" says
which direction is good. "Reference points" give the value of a trivial baseline or of the untrained model, and,
where the README reports one, the final result.

## VLA-JEPA (predicting what happens next in the game)

| Metric | What it measures | Range / better | Reference points |
|---|---|---|---|
| **Validation loss** (val) | Average error between the predicted and the actual future V-JEPA features of the game, 0.5-2 s ahead | lower is better | ~0.68 = predicting "nothing changes" (the value at the start of training) |
| **R² of the change** at +0.53 / +1.07 / +2.0 s | Share of *how the scene changes* that the model predicts, on held-out test sessions | 0 = no better than "nothing changes", 1 = perfect; higher is better | 0 = "nothing changes" |
| **Language loss** (language val) | How surprised the model is by held-out reasoning texts written by Glimmer (per word piece) | lower is better | ~12.2 = random guessing; Glimmer before fine-tuning 1.16 |
| Gradient norms, language weight | How hard each objective pulls on Glimmer's weights; the weight is set so both pull equally | a check, not a score | |

## SFT (learning to press the person's buttons, after reasoning)

| Metric | What it measures | Range / better | Reference points |
|---|---|---|---|
| **Validation loss** | How surprised the model is by the button the person actually pressed, on 300 held-out turns | lower is better | untrained Glimmer 3.11 |
| **Exact match, switch turns** | % of turns where the person *changed* button and the model predicts the new one exactly | 0-1, higher is better | copying the last button = 0 |
| **Exact match, repeat turns** | % of turns where the person *kept* the same button and the model predicts it | 0-1, higher is better | copying the last button = 1 |
| **Exact match, natural** | The two above weighted by how often people really switch (half of the turns here) | 0-1, higher is better | **copying the last button = 0.50** (the bar to beat) |
| **Reasoning loss** (trace val) | How surprised the model is by held-out Odysseus reasoning texts: is it writing good perception + reasoning? | lower is better | untrained Glimmer 1.16 |

## RL (GAIL: the agent plays, a discriminator compares it with sub-03)

| Metric | What it measures | Range / better | Reference points |
|---|---|---|---|
| **Discriminator accuracy** (human-likeness) | % of moves where a classifier correctly tells the agent's move from sub-03's | **0.5 = indistinguishable (the goal)**, 1.0 = trivially different; lower is better | 0.5 |
| Discriminator accuracy, matched | Same, comparing moves at the same point of a level (fairer: otherwise it mostly detects *where* Mario is) | same as above | 0.5 |
| **Action KL** | Difference between the agent's mix of buttons and the person's | 0 = identical mix; lower is better | Glimmer out of the box 3.31; after SFT 0.09 (sub-01, README) |
| Reasoning share | % of turns where the agent writes its reasoning before acting | want 1.0 | Glimmer out of the box 0.95; after SFT 1.0 (README) |
| KL to SFT | How far RL has moved the model from the SFT model it started from | small = barely changed; large = drifting (risk of losing the reasoning) | 0 at the start of RL |
| Deaths | Deaths per iteration (512 turns) | lower is better | |

## Brain prediction (sub-03's fMRI on held-out sessions)

| Metric | What it measures | Range / better | Reference points |
|---|---|---|---|
| **Gain** | Correlation between predicted and measured brain activity, averaged over 1,000 regions, **minus** a nuisance-only model (buttons, level, time, screen brightness, motion, sound) | 0 = adds nothing beyond the nuisance; higher is better | V-JEPA 2.1 alone +0.0686; V-JEPA + the agent's state + its reply context +0.0760 (README) |
| Top-100 gain | Same, on the 100 best-predicted regions | higher is better | |
| Per network | The gain within each brain network (visual, motor, attention, salience, control, default mode) | higher is better | `brain_pred/encoding/region_report.py` |
