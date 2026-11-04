#!/bin/bash
# Download the Odysseus VLM (Odysseus-Project/Odysseus, Qwen3-VL-8B fine-tune, Apache-2.0, ~17.5 GB) into
# $MJG_MODELS/Odysseus, and the Odysseus prompt files into experiments/odysseus/upstream, on a compute node.
# The model is gated ("accept conditions" on huggingface.co); the HF account of the token in
# $HF_HOME must have accepted them first.  Submit from the code folder:
#   sbatch experiments/odysseus/download.sbatch
#SBATCH --job-name=odysseus-dl
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/odysseus_download_%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"   # HF_HOME (holds the token), MJG_MODELS, MJG_MUSE_PY
unset HF_HUB_OFFLINE
DEST=$MJG_MODELS/Odysseus
echo "downloading to $DEST"
"$MJG_MUSE_PY" - <<PYEOF
from huggingface_hub import snapshot_download
p = snapshot_download("Odysseus-Project/Odysseus", local_dir="$DEST")
print("downloaded to", p)
PYEOF
du -sh "$DEST"
H=$MJG_CODE/experiments/odysseus/upstream
mkdir -p "$H"
for f in verl/verl/utils/game_envs/super_mario_land/env_prompts.py \
         verl/verl/experimental/agent_loop/game_agent_loop.py; do
  curl -fsSL "https://raw.githubusercontent.com/odysseus-project/odysseus-mario-training/main/$f" -o "$H/$(basename $f)"
done
ls -la "$H"
echo "finished at $(date)"
