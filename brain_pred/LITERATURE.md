# Literature: imitation, predictive objectives and brain alignment in games

Each entry says **what it did** and **why it matters here**. Sections 1-3 and the first part of section 4 were
checked against the source pages; entries marked "from memory" or "not checked" should be verified before citing.

## 1. Closest to this project: video games + fMRI + imitation

**Paugam, Pinsard, St-Laurent, Lajoie & Bellec (2026). "Training neural networks from scratch in a videogame leads to brittle brain encoding." eLife reviewed preprint.**
https://elifesciences.org/reviewed-preprints/110830
- CNeuroMod *Super Mario Bros.* fMRI, 5 participants, ~15 h of fMRI each, MIST parcellation. Small CNNs trained
  from scratch on the same frames with different objectives: PPO, per-participant imitation of button presses
  (two levels held out), and a proxy that predicts ResNet152 features. Baselines were an untrained net and a
  pretrained ResNet.
- Objective differences on training levels were tiny (<1% R²), the untrained net came close,
  no layer/objective mapped preferentially to specific areas, and encoding **collapsed on
  held-out levels** for all models (PPO generalized best, then untrained, then imitation). The eLife assessment
  notes that ridge overfitting is an alternative explanation for the collapse.
- **Relevance:** the main baseline and the main motivation: pretrained models (V-JEPA 2.1, Glimmer) instead of
  from-scratch networks, tested on held-out sessions, with ridge penalties chosen on training data only
  (`mjg/encoding.py`: per-target alpha by cross-validation over training episodes).

**Kemtur et al. (2022), via Bellec, "Gamer in the scanner: imitation of human video gameplay and fMRI brain activity using artificial neural networks."**
https://zenodo.org/records/7502069
- *Shinobi III*, N=6, up to ~140 h of fMRI per subject, MRI-compatible controller.
  Agents trained to imitate human actions, and their internal representations predicted
  individual brain activity.
- **Relevance:** precedent for "imitate a person's play, then evaluate the agent against their BOLD".

**CNeuroMod data and tooling**
- Mario dataset: https://github.com/courtois-neuromod/mario
- **mario.tutorials** (CNeuroMod, 2025; https://github.com/courtois-neuromod/mario.tutorials). Notebooks: dataset
  overview, GLM (actions and game events, shinobi_fmri style), PPO agent training, ridge encoding from the PPO CNN
  layers (single session). What it offers:
  - **Low-level confounds** per frame: luminance, optical flow and audio envelope (`gamelogs/*_lowlevel.npy`,
    `dev_replays` branch of the mario dataset). This project computes the same three signals from its own replays
    (`brain_pred/features/lowlevel.py`) and adds them to the nuisance model (`encode.py --lowlevel`), so a feature
    band only gets credit for what they do not explain.
  - **Pretrained PPO agents** (`models/*.pth`, 2.5 MB each: 4 levels at 1M and 10M steps, all 22 levels at
    10M steps; 4-frame 84x84 grayscale input, 4 conv + fc512, 12 actions). These are the Paugam 2026 model
    family: a classic RL agent to compare with Glimmer's internal state.
  - Scrub regressors for TRs outside a valid play, button-press counts as a confound, event GLMs from
    `mario.annotations` (kills, hits, power-ups, coins).
  - Not followed here: a random 70/30 split of TRs within one session (autocorrelated neighbours leak) and a
    "noise ceiling" from correlating different runs (different stimuli in Mario). Held-out sessions are the
    stricter test.
- mario.scenes, which splits 23 levels into behavioral scenes and annotates .bk2 + fMRI:
  https://github.com/courtois-neuromod/mario.scenes. Useful for held-out-scene/level
  splits and for event-level analyses within TRs.
- Dataset overview: https://paper.cneuromod.ca/intro/

**Cross, Cockburn, Yue & O'Doherty (2021). "Using deep reinforcement learning to reveal how the brain encodes abstract state-space representations in high-dimensional environments." Neuron.**
https://www.gwern.net/docs/rl/2020-cross.pdf
- DQN hidden states from Atari play mapped onto human fMRI recorded during the same games.
- **Relevance:** the classic agent → BOLD encoding setup for games.

**Tomov, Tsividis, Pouncy, Tenenbaum & Gershman (2023). "The neural architecture of theory-based reinforcement learning." Neuron.**
https://www.biorxiv.org/content/10.1101/2022.06.14.496001.full.pdf
- Theory-based (model-based, object/relation-level) RL in Atari-like games, with fMRI
  signatures of theory representation and updating.
- **Relevance:** the theoretical anchor for giving the agent a world model: VLA-JEPA predicts the next seconds
  of the game in V-JEPA's latent space, a latent world model at a cognitive timescale.

## 2. Brain-tuning: training models on neural data

**Brain-tuned V-JEPA 2.1 ViT-L (Hugging Face: RAntonello/vjepa2.1-vitl-braintuned).**
https://huggingface.co/RAntonello/vjepa2.1-vitl-braintuned
- LoRA r=8 / alpha=16 on attention qkv and proj of blocks 15-20 (of 24) of V-JEPA 2.1, trained to predict
  CNeuroMod *Friends* fMRI (Schaefer-1000, ~98k 4-s clips, S1-5, subjects 1, 2, 3, 5; season 6 held out) through
  a **frozen RidgeCV readout** fit once in closed form on 40,000 frozen-feature clips, then merged. License
  CC-BY-NC-4.0. No paper is linked on the card.
- **Relevance:** a video trunk trained directly on BOLD (for movies): a comparison point for models trained
  without fMRI, and its frozen readout is one answer to the risk of a readout absorbing everything.

**Moussa, Klakow & Toneva (ICLR 2025). "Improving semantic understanding in speech language models via brain-tuning."**
https://arxiv.org/abs/2410.09230
**Follow-up (2025). "Brain-tuning improves generalizability and efficiency of brain alignment in speech models."**
https://arxiv.org/abs/2510.21520
- Fine-tuning speech models to predict fMRI improves alignment *and* downstream semantic
  tasks, and the alignment generalizes across subjects and data.
- **Relevance:** evidence that neural targets can improve task performance, not just
  alignment; their evaluation design is a template for cross-subject tests.

**Dapello et al. (ICLR 2023). "Aligning model and macaque inferior temporal cortex representations improves model-to-human behavioral alignment and adversarial robustness."**
https://www.biorxiv.org/content/10.1101/2022.07.01.498495.full.pdf
- Representational alignment to IT during training improves human behavioral alignment
  and robustness.
- **Relevance:** the strongest precedent that **neural alignment and human-like behavior go together**, the
  link this project draws between playing like a person and predicting their brain.

## 3. Prediction, timescale and video models

**Caucheteux, Gramfort & King (2023). "Evidence of a predictive coding hierarchy in the human brain listening to speech." Nature Human Behaviour.**
https://pmc.ncbi.nlm.nih.gov/articles/PMC10038805
- Adding long-range, multi-timescale *forecast* representations to LM activations improves
  brain mapping, and the optimal forecast distance grows along the cortical hierarchy.
- **Relevance:** the language analogue of predicting the game ahead; it predicts that the best prediction
  horizon should differ between brain regions.

**"Self-supervised predictive learning accounts for cortical layer-specificity" (Nature Communications, 2025).**
https://www.nature.com/articles/s41467-025-61399-5
- Predictive self-supervised objectives reproduce layer-specific cortical properties.
- **Relevance:** supports predictive (JEPA-style) objectives as brain-like learning rules.

**"One hundred neural networks and brains watching videos: lessons from alignment" (bioRxiv 2024, ICLR 2025).**
https://www.biorxiv.org/content/10.1101/2024.12.05.626975v3
- 99 models vs. Bold Moments fMRI. Temporal (video) modeling helps mainly **early visual**
  areas; action-recognition training helps later areas, mostly at the classifier layer.
  The authors note that a controlled comparison of training objectives with matched
  architectures is missing.
- **Relevance:** suggests looking for what an agent adds outside early visual cortex, where video features
  alone do less.

**VIBE: Video-Input Brain Encoder for fMRI response modeling (2025).**
https://arxiv.org/abs/2507.17958
**"The human brain as a dynamic mixture of expert models in video understanding" (ICLR 2026).**
https://www.biorxiv.org/content/10.1101/2025.10.27.684803v1.full.pdf
- Recent video → fMRI encoders (Algonauts-era).
- **Relevance:** best practice for the encoding pipeline and strong encoding baselines to report against.

**V-JEPA 2.1: Unlocking dense features in video self-supervised learning (2026).**
https://arxiv.org/abs/2603.14482
- The video encoder: VLA-JEPA's prediction targets and a brain-encoding feature.

## 4. Reasoning models, auxiliary prediction and games

**Oota, Khandelwal, Pahwa, Namburi, Chakraborty, Raju & Gupta (2026). "Brain alignment of reasoning and action
representations from vision-language and action models during naturalistic gameplay." arXiv 2605.19352.**
https://www.emergentmind.com/papers/2605.19352
- Six Atari-style games, the Tomov et al. (2023) fMRI data (32 participants, 11 ROIs). Vision-language models
  (Qwen2.5-VL-7B, InternVL3-8B) and large action models beat both RL baselines (theory-based EMPA, DDQN) at predicting
  brain activity, even at matched dimensionality; prompting increases alignment, most in frontal / parietal / motor
  areas (~2x early visual). Thinking-mode reasoning traces were reported as less aligned than action outputs, but the
  reported numbers look identical, so that comparison is unclear.
- **Relevance:** the closest precedent to this pipeline (a VLM that plays, its reasoning as brain features). It is
  on Atari + Tomov's data, not CNeuroMod Mario, and it does not train the model to play like the person. Its
  variance partitioning (action vs reasoning) applies to the reasoning-trace encoding here.

**Fang & Stachenfeld (2023). "Predictive auxiliary objectives in deep RL mimic learning in the brain." arXiv 2310.06089.**
https://arxiv.org/html/2310.06089v1
- Predictive auxiliary objectives in deep RL; longer-horizon prediction transfers better to new environments; the
  auxiliary model is related to hippocampal learning.
- **Relevance:** direct support for an auxiliary prediction objective (VLA-JEPA) and for predicting several
  horizons ahead.

**Not checked against the papers:**
- **"Reason to Play: behavioral and brain alignment" (2026), arXiv 2605.08019, https://botcs.github.io/reason-to-play/.**
  Humans (32 fMRI participants) and frontier reasoning models learn simple VGDL games without rules; the models'
  hidden states predict BOLD during game learning, mostly through game-state structure. With Oota et al. above,
  both use the Tomov-lab game data (Atari-style / VGDL), off-the-shelf models, and slow, puzzle-like games. Its
  banded ridge (one penalty per feature band) is the idea behind `mjg.encoding.fit_episodes_banded`.
- **Threat imminence in games (human fMRI only):** a modified PAC-MAN with simultaneous threat and reward (Human
  Brain Mapping 2024, doi 10.1002/hbm.26648: a distributed threat-imminence system); a predator-prey maze game
  (2025: imminence predicts cinguloinsular and dorsal frontoparietal activity); the classic "virtual predator"
  (Mobbs et al.). None compares the human threat response with an artificial agent's.

**From memory (verify before citing):**
- Hasson, Yang, Vallines, Heeger & Rubin (2008, J. Neurosci.). *A hierarchy of temporal receptive windows in human
  cortex.* Integration windows grow from sensory to higher areas: another reason to expect the best horizon or
  pooling window to differ by region.
- Huth et al. (2016, Nature), semantic maps: features brought to the TR with a Lanczos (anti-aliasing) filter, the
  standard alternative to plain mean-pooling of features over a TR.

## 5. Earlier neural regularization (from memory, not verified)

- Schwartz, Tang & Wehbe (NeurIPS 2019). *Inducing brain-relevant bias in natural language
  processing models.* Fine-tuned BERT on fMRI/MEG from reading.
- Li et al. (NeurIPS 2019). *Learning from brains how to regularize machines.* A
  mouse-V1 similarity regularizer improved CNN robustness.
- Fong, Scheirer & Cox (Sci. Reports 2018). *Using human brain activity to guide machine
  learning.* fMRI-derived sample weights for classifier training.
- Bardes et al. (2024). *Revisiting feature prediction for learning visual representations
  from video* (V-JEPA 1).

## Where this project differs

CNeuroMod Mario (a fast visuomotor game, ~15 h of fMRI per person) instead of slow, puzzle-like games; a VLM
trained to play like the person (SFT on its own reasoning, then RL toward that person's play) rather than used off
the shelf; a vision baseline (V-JEPA) on the same scans; and the danger test on one objective risk scale for humans
and agents. No work found uses a **self-supervised predictive objective** as an auxiliary loss for an agent that
**imitates a person** and then tests how well the agent's internal state predicts that person's brain.
