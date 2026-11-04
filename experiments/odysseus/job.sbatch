#!/bin/bash
# Odysseus plays Super Mario Bros on our emulator (play.py): their cross-game setup adjusted to our case.
# Settings (environment): RUNS (8), BATCH (16), MAXTURNS (1000), LEVELS (default: all 22), OUTDIR, NSHARDS.
# Submit from the code folder:
#   sbatch experiments/odysseus/job.sbatch
#   NSHARDS=4 sbatch --array=0-3 experiments/odysseus/job.sbatch   (episodes split over 4 GPUs)
#SBATCH -J odysseus-play
#SBATCH --export=ALL
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH --constraint=vr40g
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/odysseus_play_%A_%a.out
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export ODY_WORKER_PY=$MJG_PY   # emulator workers: stable-retro (same Python version as the model env)
# NSHARDS must be the TOTAL number of shards (not taken from the array size, so a partial re-run of some array
# tasks still splits the episodes the same way): required for array jobs.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -z "${NSHARDS:-}" ]; then
  echo "set NSHARDS= to the total number of shards, e.g. NSHARDS=4 sbatch --array=0-3 ..."
  exit 1
fi
ARGS=(--runs "${RUNS:-8}" --batch "${BATCH:-16}" --max-turns "${MAXTURNS:-1000}"
      --shard "${SLURM_ARRAY_TASK_ID:-0}" --nshards "${NSHARDS:-1}")
if [ -n "${LEVELS:-}" ]; then
  ARGS+=(--levels $LEVELS)
fi
if [ -n "${OUTDIR:-}" ]; then
  ARGS+=(--out "$OUTDIR")
fi
"$MJG_MUSE_PY" -u experiments/odysseus/play.py "${ARGS[@]}"
