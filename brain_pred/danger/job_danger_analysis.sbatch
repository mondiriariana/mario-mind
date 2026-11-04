#!/bin/bash
# Danger experiment, step 3c for one agent (CPU): its scenario games -> generated brain (internal state only, best
# layer) -> danger analysis -> figure.
#   sbatch --dependency=afterok:<scenarios job> brain_pred/danger/job_danger_analysis.sbatch <tag>
# Output: results/danger/agent_<tag>_sub-03.json (+ _parcels.npz) and danger_agent_<tag>_sub-03.png.
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o logs/danger_%x_%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
TAG=$1
$MJG_PY -u brain_pred/danger/agent_response.py --stage prep --tag $TAG --subject sub-03 --depth -1
$MJG_SCI_PY -u brain_pred/danger/agent_response.py --stage analyze --tag $TAG --subject sub-03 --depth -1
$MJG_PY brain_pred/danger/plot_danger.py --src agent_$TAG --subject sub-03 \
  --out $MJG_ROOT/results/danger/danger_agent_${TAG}_sub-03.png
