#!/bin/bash
# Compact copy of the V-JEPA 2.1 G features, then its verification (scripts/features/compact_vjepa21G.py). CPU only.
# Deletes nothing. Submit from the code folder: sbatch slurm/compact_vjepa21G.sbatch
#SBATCH -J vjepa21G-compact
#SBATCH -p short
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/compact_vjepa21G_%j.out
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
set -euo pipefail
"$MJG_SCI_PY" -u scripts/features/compact_vjepa21G.py --procs 16
"$MJG_SCI_PY" -u scripts/features/compact_vjepa21G.py --verify --procs 16
