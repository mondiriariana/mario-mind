#!/bin/bash
# Dense V-JEPA features for every screen of every play (scripts/features/encode_frames.py), one GPU per job.
# 32 jobs, at most 12 running at once (the %12) so the GPU nodes stay available to others.
# Resumable: plays already written are skipped; each job stops before OUT exceeds MAX_TB.
# Optional: OUT, ENCODER (default V-JEPA 2.1 ViT-G), SUBJECTS, MAX_TB, BATCH, PLAYS (only these play ids).
# Submit from the code folder: sbatch slurm/encode_frames.sbatch
#SBATCH -J vjepa-frames
#SBATCH --export=ALL
#SBATCH -p medium
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/vjepa_frames_%A_%a.out
#SBATCH --array=0-31%12
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
"${MJG_VISION_PY:?set MJG_VISION_PY in slurm/paths.local.sh}" scripts/features/encode_frames.py --out ${OUT:-$MJG_ROOT/features/vjepa21G_frames} \
  --encoder "${ENCODER:-vjepa2.1:\${MJG_VJEPA21_G}}" --subjects ${SUBJECTS:-sub-01 sub-02 sub-03 sub-05 sub-06} \
  --shard $SLURM_ARRAY_TASK_ID --nshards ${SLURM_ARRAY_TASK_COUNT:-32} --max-tb ${MAX_TB:-4.2} \
  --batch ${BATCH:-32} ${PLAYS:+--plays $PLAYS}
