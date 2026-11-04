#!/bin/bash
# RL (GAIL) of the Glimmer agent on 4 GPUs: rl.py with the settings of rl.sh (vLLM generates the rollouts on two
# GPUs, the HF model computes log-probs and updates the LoRA on the other two).
# Inputs (see rl.sh): RL_MODEL = the VLA-JEPA Glimmer model folder, RL_INIT = the SFT LoRA.
# Output: $MJG_ROOT/results/glimmer_imitation/rl/<tag>/ (log.json, lora_*.pt, state.pt, rollouts/).
#   RL_MODEL=<VLA-JEPA glimmer dir> RL_INIT=<SFT lora.pt> sbatch experiments/glimmer_imitation/job_rl.sbatch
#SBATCH -J rl
#SBATCH -p short
#SBATCH --gres=gpu:h200:4
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 12:00:00
#SBATCH -o logs/rl_%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ODY_WORKER_PY=$MJG_PY
source experiments/glimmer_imitation/rl.sh   # sets MODEL, INIT, ALGO, ARGS
echo "$(date) RL: $ALGO from $INIT on $MODEL; ${ARGS[*]}"
"$MJG_MUSE_PY" -u experiments/glimmer_imitation/rl.py --algo "$ALGO" --model-dir "$MODEL" --init "$INIT" "${ARGS[@]}"
