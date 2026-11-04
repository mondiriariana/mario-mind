#!/bin/bash
# Download a whole Hugging Face repository (many weight files), file by file, as Slurm jobs (this script only
# submits them): an array of jobs shares the files (fetch_hf_repo.sbatch), each resumes partial downloads and
# checks every weight file against Hugging Face's sha256; a final job lists anything missing
# (fetch_hf_check.sbatch). Gated repos need approved access and a read token in $HF_HOME/token.
#   bash slurm/fetch_hf_repo.sh <repo id> <destination folder> [<N jobs, default 32>]
#   bash slurm/fetch_hf_repo.sh meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
#       "$MJG_MODELS/Muse-Spark-Open-816B-A42B-NVFP4" 32
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${1:?repo id}
DEST=${2:?destination folder}
N=${3:-32}
mkdir -p "$DEST/.fetch" logs
a=$(sbatch --parsable --array=0-$((N - 1)) --export=ALL,REPO=$REPO,DEST=$DEST,N=$N slurm/fetch_hf_repo.sbatch)
c=$(sbatch --parsable --dependency=afterany:$a --export=ALL,REPO=$REPO,DEST=$DEST slurm/fetch_hf_check.sbatch)
echo "$REPO -> $DEST: $N download jobs $a, final check job $c"
