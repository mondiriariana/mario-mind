#!/bin/bash
# Calibration part 1 (CPU): cut every fMRI play of a subject into the agent's turns (build_scan_turns.py).
#   sbatch brain_pred/agent/job_scan_turns.sbatch sub-03 [--all-lives]
#SBATCH -J scan-turns
#SBATCH -p short
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/scan_turns_%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
"$MJG_PY" -u brain_pred/agent/build_scan_turns.py --subject "$1" --procs 14 "${@:2}"
