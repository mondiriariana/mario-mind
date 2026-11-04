#!/bin/bash
# VLA-JEPA: full fine-tune of Glimmer 30B on 4 H200 GPUs (torchrun, FSDP2), with the settings in long_run.sh.
# Submit from the code folder:
#   sbatch experiments/glimmer_vlajepa/job_long.sbatch
# The run checkpoints and exits before the job's time limit; continue it with the same settings:
#   RESUME=1 sbatch experiments/glimmer_vlajepa/job_long.sbatch
#SBATCH -J glim-long
#SBATCH -p medium
#SBATCH --gres=gpu:h200:4
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/glim_long_%j.out
set -euo pipefail
source slurm/env.sh
export TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
source experiments/glimmer_vlajepa/long_run.sh   # sets ARGS and RESUME
if [ "$RESUME" = 1 ]; then
  ARGS+=(--resume)
fi
echo "$(date) VLA-JEPA: ${ARGS[*]}"
"$MJG_MUSE_PY" -u -m torch.distributed.run --standalone --nproc-per-node 4 \
  experiments/glimmer_vlajepa/train.py "${ARGS[@]}"
