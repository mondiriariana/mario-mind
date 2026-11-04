#!/bin/bash
# Full NES RAM for every human play (scripts/data/extract_full_ram.py), CPU only, 32 jobs. Plays already
# written are skipped. Submit from the code folder:
#   sbatch slurm/extract_full_ram.sbatch
#SBATCH -J ram-full
#SBATCH --export=ALL
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 04:00:00
#SBATCH -o logs/ram_full_%A_%a.out
#SBATCH --array=0-31
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
"$MJG_PY" scripts/data/extract_full_ram.py --shard "$SLURM_ARRAY_TASK_ID" --nshards "${SLURM_ARRAY_TASK_COUNT:-32}"
