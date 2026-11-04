#!/bin/bash
# Danger direction (CPU): danger readout vs brain-danger readout per model and layer (danger_direction.py).
#   sbatch brain_pred/danger/job_direction.sbatch --subject sub-03 --tags sft plain [danger_direction.py args]
#SBATCH -J danger-dir
#SBATCH -p short
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/danger_dir_%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export MJG_TMP=$MJG_SCRATCH/tmp/$SLURM_JOB_ID  # per job, also when the submitter's environment is exported
export TMPDIR=$MJG_TMP
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
$MJG_SCI_PY -u brain_pred/danger/danger_direction.py "$@"
