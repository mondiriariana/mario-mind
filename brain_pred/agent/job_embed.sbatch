#!/bin/bash
# Calibration part 2 (GPU): the agent's internal state at every scan (embed_scans.py). Array over run parts.
#   sbatch --array=0-3 brain_pred/agent/job_embed.sbatch <subject> <tag> <lora.pt|none> [extra embed_scans.py args]
#   e.g. ... sub-03 sft <SFT lora.pt> --model-dir <VLA-JEPA Glimmer dir> --player sub-03
#SBATCH -J agent-embed
#SBATCH -p medium
#SBATCH --gres=gpu:l40s:2
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 08:00:00
#SBATCH -o logs/agent_embed_%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N=${SLURM_ARRAY_TASK_COUNT:-1}
I=${SLURM_ARRAY_TASK_ID:-0}
LORA=""
if [ "${3:-none}" != none ]; then
  LORA="--lora $3"
fi
"$MJG_MUSE_PY" -u brain_pred/agent/embed_scans.py --subject "$1" --tag "$2" $LORA --part $I/$N \
  ${EMBED_BATCH:+--batch $EMBED_BATCH} "${@:4}"
