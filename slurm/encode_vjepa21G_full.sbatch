#!/bin/bash
# V-JEPA 2.1 ViT-G features of every human play (scripts/features/encode_frames.py), one GPU per job:
#   one sample every 8 screens (7.5 per s); each clip = 16 screens 8 frames apart (the last ~2.1 s, past only)
#   <play>.npy       float16 [n, 4 depths (11,23,37,47), 8x8 grid, 1664]  time-averaged over the clip
#   <play>.last.npy  float16 [n, 8x8, 1664]  final block, most recent tubelet only
#   <play>.game.npz  sampled frame indices + buttons and RAM variables for every frame
# About 2.4 TB for the 5 subjects. 32 jobs, at most 12 running at once; plays with a .json are skipped.
# Needs MJG_VISION_PY: a Python with stable-retro, torch >= 2.6, torchvision and timm (for Meta's V-JEPA 2.1 code).
# Submit from the code folder: sbatch slurm/encode_vjepa21G_full.sbatch
#SBATCH -J vjepa21G-full
#SBATCH --export=ALL
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH --constraint=vr40g
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/vjepa21G_full_%A_%a.out
#SBATCH --array=0-31%12
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
"${MJG_VISION_PY:?set MJG_VISION_PY in slurm/paths.local.sh}" scripts/features/encode_frames.py \
  --out ${OUT:-$MJG_ROOT/features/vjepa21G_c16s8_f8_g8} \
  --encoder "vjepa2.1:${MJG_VJEPA21_G}" --subjects sub-01 sub-02 sub-03 sub-05 sub-06 \
  --shard $SLURM_ARRAY_TASK_ID --nshards ${SLURM_ARRAY_TASK_COUNT:-32} \
  --clip-len 16 --clip-stride 8 --frame-step 8 --grid 8 --save-last --save-game --batch 8 --max-tb 3.0
