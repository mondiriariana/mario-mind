#!/bin/bash
# Danger experiment, step 3b (GPU): one agent plays every scenario (scenarios.py) from the human moment on, with its
# internal state saved (glimmer_play --scenarios --save-hidden). 4 L40S: two players of 2 GPUs, half the scenarios
# each; with --gres=gpu:l40s:2 on the command line: one player, all scenarios.
#   sbatch -J scen-sft   brain_pred/danger/job_scenarios.sbatch sft   <SFT lora.pt> [extra glimmer_play args]
#   sbatch -J scen-plain brain_pred/danger/job_scenarios.sbatch plain none
# Env: SCEN (scenario file, default results/danger/scenarios_v1.json), TURNS (80), SCEN_BATCH (12).
# Output: results/danger/agent_<tag>/<scenario>/run-000/ (steps.jsonl, inputs.npy, meta.json, hidden.npy).
# Resumable: finished scenarios are skipped.
#SBATCH -p medium
#SBATCH --gres=gpu:l40s:4
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 12:00:00
#SBATCH -o logs/scenarios_%x_%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export MJG_TMP=$MJG_SCRATCH/tmp/$SLURM_JOB_ID  # per job, also when the submitter's environment is exported
export TMPDIR=$MJG_TMP
mkdir -p "$TMPDIR"
export TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ODY_WORKER_PY=$MJG_PY
TAG=$1
LORA=$2
shift 2
R=$MJG_ROOT/results/danger
SC=${SCEN:-$R/scenarios_v1.json}
PY=$MJG_MUSE_PY
$PY - "$SC" "$TMPDIR" <<'EOF'
import json, sys
sc = json.load(open(sys.argv[1]))
for h in (0, 1):
    json.dump(sc[h::2], open(f"{sys.argv[2]}/scen_{h}.json", "w"))
EOF
L=""
if [ "$LORA" != none ]; then
  L="--lora $LORA"
fi
A="--save-hidden --max-turns ${TURNS:-80} --max-new 768 --batch ${SCEN_BATCH:-12} --seed 0 --out $R/agent_$TAG $L $*"
NG=$(nvidia-smi -L | wc -l)
GP=experiments/glimmer_imitation/glimmer_play.py
LOG=logs/scenarios_${TAG}_${SLURM_JOB_ID}
if [ "$NG" -ge 4 ]; then  # two players of 2 GPUs, half the scenarios each
  CUDA_VISIBLE_DEVICES=0,1 $PY -u $GP --scenarios $TMPDIR/scen_0.json $A > ${LOG}_a.out 2>&1 &
  CUDA_VISIBLE_DEVICES=2,3 $PY -u $GP --scenarios $TMPDIR/scen_1.json $A > ${LOG}_b.out 2>&1 &
  wait
else                      # one player on 2 GPUs, all scenarios
  $PY -u $GP --scenarios $SC $A > ${LOG}_a.out 2>&1
fi
echo "$TAG: $(ls $R/agent_$TAG/*/run-000/hidden.npy 2>/dev/null | wc -l) scenarios with hidden.npy"
