#!/bin/bash
# Brain encoding, CPU only: per-scan features, then encoding models (build_tr_features.py, encode.py).
#   sbatch brain_pred/encoding/job.sbatch sub-01 [sub-02 ...]
#SBATCH -J brain-enc
#SBATCH -p short
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o logs/brain_enc_%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
"$MJG_SCI_PY" -u brain_pred/encoding/build_tr_features.py --subjects "$@" --procs 12
"$MJG_SCI_PY" -u brain_pred/encoding/encode.py --subjects "$@"
