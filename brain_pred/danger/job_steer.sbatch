#!/bin/bash
# Danger steering (GPU): the SFT model (VLA-JEPA Glimmer + SFT LoRA) plays the enemy + gap danger scenarios with
# alpha x its danger direction added at one layer (steer_play.py). One strength per job, 2 L40S.
#   sbatch -J steer-p4 --export=ALL,MODEL=<model dir>,LORA=<SFT lora.pt> brain_pred/danger/job_steer.sbatch \
#     <alpha> [dir npz] [extra glimmer_play args]
# Output: results/danger/steer/<OUTTAG><KIND>_a<alpha>/<scenario>/run-000/ (steps.jsonl, meta.json, hidden.npy).
# Resumable. <alpha> may be a comma list (run one after the other in this job).
# Env (pass with sbatch --export=ALL,...): MODEL (required: the VLA-JEPA Glimmer folder), LORA (required: the SFT LoRA),
# KIND (pattern), TURNS (20), TYPES ("enemy hole"), SCEN_BATCH (12), MAXSCEN (0 = all), PLAYER (sub-03), OUTTAG (prefix
# of the output folder, e.g. <tag>_; steer_analyze.py --prefix reads <OUTTAG><KIND>_a*). steer_model.sh runs the whole
# chain (direction -> steering -> analysis) for one agent.
#SBATCH -p short
#SBATCH --gres=gpu:l40s:2
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 06:00:00
#SBATCH -o logs/steer_%x_%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export MJG_TMP=$MJG_SCRATCH/tmp/$SLURM_JOB_ID  # per job, also when the submitter's environment is exported
export TMPDIR=$MJG_TMP
mkdir -p "$TMPDIR"
export TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ODY_WORKER_PY=$MJG_PY
ALPHA=$1
R=$MJG_ROOT/results
DIR=${2:-$R/danger/direction/sub-03_agent_sft_d1_danger_dir.npz}
shift
if [ $# -gt 0 ]; then
  shift
fi
KIND=${KIND:-pattern}
MODEL=${MODEL:?path to the VLA-JEPA Glimmer model folder}
LORA=${LORA:?path to the SFT LoRA (lora_step<N>.pt)}
echo "model $MODEL + $LORA, direction $DIR"
if [ ! -f "$LORA" ] || [ ! -f "$DIR" ]; then
  echo "missing LoRA or direction; exiting"
  exit 1
fi
PY=$MJG_MUSE_PY
$PY - "$R/danger/scenarios_v1.json" "$TMPDIR/scen.json" "${TYPES:-enemy hole}" <<'PYEOF'
import json
import os
import sys

types = sys.argv[3].split()
sc = []
for s in json.load(open(sys.argv[1])):
    if s["type"] in types:
        sc.append(s)
n = int(os.environ.get("MAXSCEN", "0") or 0)
if n:  # balanced: the first n/2 of each type
    balanced = []
    for t in types:
        of_type = []
        for x in sc:
            if x["type"] == t:
                of_type.append(x)
        balanced.extend(of_type[:max(1, n // 2)])
    sc = balanced
json.dump(sc, open(sys.argv[2], "w"))
print(len(sc), "scenarios")
PYEOF
for A in ${ALPHA//,/ }; do
  O=$R/danger/steer/${OUTTAG:-}${KIND}_a${A}
  $PY -u brain_pred/danger/steer_play.py --dir $DIR --alpha $A --dir-kind $KIND -- \
    --scenarios $TMPDIR/scen.json --model-dir $MODEL --lora $LORA --player ${PLAYER:-sub-03} --save-hidden \
    --max-turns ${TURNS:-20} --max-new 768 --batch ${SCEN_BATCH:-12} --seed 0 --out $O "$@"
  echo "alpha $A: $(ls $O/*/run-000/meta.json 2>/dev/null | wc -l) scenarios done"
done
