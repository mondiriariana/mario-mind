#!/bin/bash
# Moment index + coverage (scripts/data/build_moment_index.py): the moments VLA-JEPA's frame set is drawn from.
# CPU only. Run after slurm/extract_full_ram.sbatch has finished. Submit from the code folder:
#   sbatch slurm/build_moment_index.sbatch
#SBATCH -J moment-index
#SBATCH --export=ALL
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/moment_index_%j.out
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
"$MJG_PY" scripts/data/build_moment_index.py --out "$MJG_ROOT/results/moment_index/v1"
