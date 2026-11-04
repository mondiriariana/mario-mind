#!/bin/bash
# Row masks (gameplay / in_control) for the V-JEPA 2.1 G features (scripts/features/build_row_masks.py), CPU only,
# with the timer check on 20 plays against features/ram_full. Submit from the code folder:
#   sbatch slurm/build_row_masks.sbatch        # optional: FEAT=<feature folder>
#SBATCH -J row-masks
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/build_row_masks_%j.out
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
"$MJG_PY" -u scripts/features/build_row_masks.py --feat-dir "${FEAT:-$MJG_ROOT/features/vjepa21G_c16s8_f8_g8}" \
  --check 20
