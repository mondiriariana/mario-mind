# Site-specific paths. Copy to slurm/paths.local.sh (git-ignored) and fill in.
# Every launcher sources slurm/env.sh, which reads this file; the Python scripts read the same variables.

# Storage
export MJG_ROOT=/path/to/project/mario-jepa-gail      # code/, runs/, features/, results/
export MJG_SCRATCH=/path/to/scratch/mario-jepa-gail   # fast temporary space: replayed frames, scan turns, job temp
export MJG_MODELS=/path/to/models                     # Muse-Glimmer-30B/, Odysseus/, ...

# CNeuroMod data
export MJG_CNEUROMOD=/path/to/cneuromod
export MJG_DATA=$MJG_CNEUROMOD/data/mario             # .bk2 replays (+ mp4) of the mario dataset
export MJG_ANNOT=$MJG_CNEUROMOD/data/mario.annotations
export MJG_INTEGRATION=$MJG_CNEUROMOD/mario.stimuli   # folder holding SuperMarioBros-Nes/ (ROM + states)
export MJG_ATLAS=$MJG_CNEUROMOD/atlas                 # Schaefer-1000 labels (tpl-MNI152NLin2009cAsym_..._dseg.tsv)

# Model weights
export HF_HOME=/path/to/huggingface
export MJG_VJEPA21_G=/path/to/models/vjepa2_1-vitG-384     # V-JEPA 2.1 ViT-G (2B, 384 px): checkpoint + code/
export MJG_VJEPA_G=/path/to/models/vjepa2-vitg-fpc64-256   # V-JEPA 2 ViT-G (optional)

# Python environments
export MJG_PY=/path/to/envs/gym/bin/python         # emulator: stable-retro, numpy, opencv
export MJG_MUSE_PY=/path/to/envs/muse/bin/python   # model: torch (CUDA 12.8), transformers, peft, accelerate, pandas, scikit-learn
export MJG_VLLM_PY=/path/to/envs/vllm/bin/python   # vLLM server for RL rollouts
export MJG_VISION_PY=/path/to/envs/vision/bin/python # V-JEPA features: stable-retro, torch >= 2.6, torchvision, timm
export MJG_SCI_PY=/path/to/envs/sci/bin/python     # analysis: scikit-learn, nilearn, h5py, pandas
