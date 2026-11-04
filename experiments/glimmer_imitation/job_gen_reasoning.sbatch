#!/bin/bash
# gen_human_reasoning.py on 4x L40S (two vLLM servers). Submit from the code folder:
#   sbatch experiments/glimmer_imitation/job_gen_reasoning.sbatch --lora <LoRA .pt> --data-dirs <...> --out <jsonl>
#SBATCH -J gen-reasoning
#SBATCH -p medium
#SBATCH --gres=gpu:l40s:4
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 03:00:00
#SBATCH -o logs/gen_reasoning_%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export TRANSFORMERS_OFFLINE=1
"$MJG_MUSE_PY" -u experiments/glimmer_imitation/gen_human_reasoning.py "$@"
