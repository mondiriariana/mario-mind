#!/bin/bash
# SFT of Glimmer (train_sft.py) with the settings in sft.sh, on 2x L40S.
# Input (see sft.sh): SFT_MODEL = the VLA-JEPA Glimmer model folder.
# Output: $MJG_ROOT/results/glimmer_imitation/sft/<tag>/ (lora_step<N>.pt, log.json, train_state.pt).
#   SFT_MODEL=<VLA-JEPA Glimmer dir> sbatch experiments/glimmer_imitation/job_sft.sbatch
#SBATCH -J sft
#SBATCH -p medium
#SBATCH --gres=gpu:l40s:2
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 07:00:00
#SBATCH -o logs/sft_%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source experiments/glimmer_imitation/sft.sh   # sets MODEL and ARGS
echo "SFT on $MODEL: ${ARGS[*]}"
"$MJG_MUSE_PY" -u experiments/glimmer_imitation/train_sft.py --model-dir "$MODEL" "${ARGS[@]}"
