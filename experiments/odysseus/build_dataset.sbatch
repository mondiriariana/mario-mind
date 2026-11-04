#!/bin/bash
# Odysseus SFT dataset (build_dataset.py): filters, de-duplication, split, screens replayed from inputs.npy.
# Run after all episodes are done. Submit from the code folder (extra arguments go to build_dataset.py):
#   sbatch experiments/odysseus/build_dataset.sbatch            # e.g. --check: counts only, no files
#SBATCH -J ody-dataset
#SBATCH --export=ALL
#SBATCH -p short
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/ody_dataset_%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
"$MJG_PY" -u experiments/odysseus/build_dataset.py "$@"
