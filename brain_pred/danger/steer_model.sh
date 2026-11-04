#!/bin/bash
# Danger steering for one agent, queued as a chain of three Slurm jobs (this script only calls sbatch):
#   direction (CPU, danger_direction.py on <run>.agent_<tag>.npy: that agent's own per-scan states)
#   -> steering games (2 L40S, job_steer.sbatch: that agent plays the 30 enemy + gap scenarios at each strength)
#   -> analysis (CPU, steer_analyze.py: writing and play vs strength 0)
#      -> results/danger/steer/summary_<tag>_pattern.json
#   bash brain_pred/danger/steer_model.sh <tag> <model dir> <SFT lora.pt> [alphas, default -4,0,2,4]
# The states come from embed_scans.py with the same --tag; AFTER=<job id> makes the chain wait on that job, otherwise
# they must already exist. The direction is the agent's own (its layer 26, covariance pattern), so steering strengths
# are in SD of that agent's natural spread along it.
set -euo pipefail
source slurm/env.sh
TAG=$1
MD=$2
LORA=$3
ALPHAS=${4:--4,0,2,4}
R=$MJG_ROOT/results
DEP=${AFTER:+--dependency=afterok:$AFTER}
PRE=${TAG}_
if [ ! -f "$LORA" ] && [ -z "${AFTER:-}" ]; then
  echo "no LoRA at $LORA"
  exit 1
fi
DIR=$R/danger/direction/sub-03_agent_${TAG}_d1_danger_dir.npz
D=$(sbatch --parsable $DEP -J dir-$TAG brain_pred/danger/job_direction.sbatch --subject sub-03 --tags $TAG \
  --save-dirs agent_${TAG}_d1 agent_${TAG}_d0)
S=$(sbatch --parsable --dependency=afterok:$D -J steer-$TAG \
  --export=ALL,MODEL=$MD,LORA=$LORA,OUTTAG=$PRE brain_pred/danger/job_steer.sbatch $ALPHAS $DIR)
A=$(sbatch --parsable --dependency=afterok:$S -J steerA-$TAG -p short -c 4 --mem=16G -t 01:00:00 \
  -o logs/steer_analyze_${TAG}_%j.out \
  --wrap="$MJG_SCI_PY -u brain_pred/danger/steer_analyze.py --kind pattern --prefix $PRE --dir $DIR")
echo "$TAG: direction $D -> steering $S ($ALPHAS) -> analysis $A"
echo "direction file $DIR, games results/danger/steer/${PRE}pattern_a*"
