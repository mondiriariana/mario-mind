#!/bin/bash
# The SFT agent (VLA-JEPA Glimmer + its SFT LoRA, playing as sub-03) on every scan of sub-03, on the all-lives turns
# (build_scan_turns.py --all-lives). Two passes, each as two processes per 4-GPU node (GPUs 0,1 and 2,3):
#   MODE=replies  embed_scans.py --reason --replies-only --no-state: the agent writes its reply at every decision
#                 (vLLM, 2 GPUs) -> <run>.agent_<tag>.replies.jsonl
#   MODE=states   embed_scans.py (no --reason): its prompt state at every decision (HF, 2 GPUs) -> <run>.agent_<tag>.npy
# --pool: any number of workers (jobs, nodes) share the runs through lock files. The replies come first:
# context_features.py needs only them. PART0=<i> spreads the starting points of several nodes (default 0; parts
# PART0, PART0+1 of NP, default 4). NPROC=1: one process (a 2-GPU job). TAG: output tag (default sft).
# EXTRA=...: more embed_scans.py arguments.
#   LORA=<SFT lora.pt> MODEL=<VLA-JEPA Glimmer dir> MODE=replies bash brain_pred/agent/traces_lm.sh
set -uo pipefail
source slurm/env.sh
export TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=${TAG:-sft}
LORA=${LORA:?path to the SFT LoRA (lora_step<N>.pt)}
MODEL=${MODEL:?path to the VLA-JEPA Glimmer model folder (glimmer_step<N>)}
MODE=${MODE:-replies}
if [ "$MODE" = replies ]; then
  M=(--reason --replies-only --no-state)
else
  M=()
fi
P0=${PART0:-0}
NP=${NP:-4}
for i in $(seq 0 $((${NPROC:-2} - 1))); do
  CUDA_VISIBLE_DEVICES=$((2 * i)),$((2 * i + 1)) "$MJG_MUSE_PY" -u brain_pred/agent/embed_scans.py \
    --subject sub-03 --tag $TAG --lora $LORA --model-dir $MODEL --player sub-03 --turns-split all_lives \
    "${M[@]}" --pool --part $((P0 + i))/$NP ${EXTRA:-} > logs/traces_${TAG}_${MODE}_${SLURM_JOB_ID:-x}_$i.out 2>&1 &
done
wait
echo "$(date) traces $MODE done"
