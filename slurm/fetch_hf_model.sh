#!/bin/bash
# Download a Hugging Face model with one large weights file, as Slurm jobs (this script only submits them):
# task 0 fetches the small files, tasks 1..N one byte-range piece each of the big file (fetch_hf_part.sbatch),
# then a final job joins the pieces and checks them against Hugging Face's sha256 (fetch_hf_join.sbatch).
# Gated models need approved access and a read token in $HF_HOME/token.
#   bash slurm/fetch_hf_model.sh <repo id> <destination folder> [<big file, default model.safetensors>] [<N, default 8>]
#   bash slurm/fetch_hf_model.sh facebook/vjepa2-vitg-fpc64-256 "$MJG_VJEPA_G" model.safetensors 8
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${1:?repo id}
DEST=${2:?destination folder}
BIG=${3:-model.safetensors}
N=${4:-8}
mkdir -p "$DEST/parts" logs
a=$(sbatch --parsable --array=0-$N --export=ALL,REPO=$REPO,DEST=$DEST,BIG=$BIG,N=$N slurm/fetch_hf_part.sbatch)
j=$(sbatch --parsable --dependency=afterok:$a --export=ALL,REPO=$REPO,DEST=$DEST,BIG=$BIG slurm/fetch_hf_join.sbatch)
echo "$REPO -> $DEST: pieces job $a, join + checksum job $j"
