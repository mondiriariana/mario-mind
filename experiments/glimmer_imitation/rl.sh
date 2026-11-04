# RL settings, sourced by job_rl.sbatch: sets MODEL, INIT, ALGO, ENGINE and ARGS (the rl.py flags).
# Required:
#   RL_MODEL   the VLA-JEPA Glimmer model folder (with model.safetensors.index.json)
#   RL_INIT    the SFT LoRA the policy starts from (lora_step<N>.pt from train_sft.py)
# Optional:
#   RL_TAG     run name (default full_gail_vlajepa); output in $MJG_ROOT/results/glimmer_imitation/rl/<tag>
#   RL_GEN_GPUS / RL_TRAIN_GPUS / RL_VLLM_TP   GPU layout: vLLM on 0,1 (tensor parallel 2), training on 2,3 by
#              default; e.g. 3 H200 = 0 / 1,2 / 1 (the 30B model fits one H200 for vLLM, not one 48 GB L40S)
#   RL_BAN=1   never sample '>>' / '>>Ċ' (a reply that closes </reasoning with '>>' tends to write no <answer>)
#   RL_FORCE=1 a reply without a valid <answer> is cut after </reasoning, '>\n<answer>' is appended and the model
#              writes the buttons (no gradient on the appended tokens)
MODEL=${RL_MODEL:?path to the VLA-JEPA Glimmer model folder}
INIT=${RL_INIT:?path to the SFT LoRA (lora_step<N>.pt)}
ALGO=gail
ENGINE=vllm
# Reward: a GAIL judge imitating sub-03, the brain-scan subject (its held-out scan sessions are not in the SFT data:
# build_sft.py excludes them). The judge learns from the human turns of the agent's game phase, 25 updates per
# iteration on the agent turns of the last 3 iterations. --kl-beta keeps the policy near the SFT LoRA.
# --patience 0: run to --iters or the time limit (a checkpoint every 5 iterations, lora_itNNN.pt).
# Malformed replies earn the lowest GAIL reward and no policy gradient (--format-reward 0 --format-mask); above 5%
# malformed, the update is skipped, the LoRA rolls back to lora_clean.pt and kl-beta doubles (--format-guard 0.05).
ARGS=(--player sub-03 --human-dir "$MJG_SCRATCH/glimmer_sft/all/sub-03" --kl-beta 0.04 --iters 300 --patience 0
      --games 32 --turns 16 --max-new 768
      --judge-steps 25 --judge-replay 3
      --format-reward 0 --format-mask --format-guard 0.05
      --tag "${RL_TAG:-full_${ALGO}_vlajepa}")
if [ "$ENGINE" = vllm ]; then
  ARGS+=(--engine vllm --gen-gpus "${RL_GEN_GPUS:-0,1}" --train-gpus "${RL_TRAIN_GPUS:-2,3}"
         --vllm-tp "${RL_VLLM_TP:-2}")
fi
if [ "${RL_BAN:-0}" = 1 ]; then
  ARGS+=(--ban-tokens '>>' '>>Ċ')
fi
if [ "${RL_FORCE:-0}" = 1 ]; then
  ARGS+=(--force-answer)
fi
# --resume: continue the tag's run from its state.pt if there is one (exits at once if the run is DONE)
ARGS+=(--resume)
