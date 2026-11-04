#!/bin/bash
# Builds the frame set VLA-JEPA trains on (screens + V-JEPA rows + labels) with build_set.py, on CPU.
# Submit from the code folder:
#   sbatch experiments/vjepa_spark_combo/job_build.sbatch [--out <dir>]
#SBATCH -J vsc-build
#SBATCH -p short
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/vsc_build_%j.out
set -euo pipefail
source slurm/env.sh
"$MJG_PY" -u experiments/vjepa_spark_combo/build_set.py "$@"
