#!/bin/bash
# Final job of slurm/fetch_hf_repo.sh: is every file of the repository present with its full size?
# Reads REPO and DEST from the environment (set by fetch_hf_repo.sh).
#SBATCH -J hf-check
#SBATCH -p short
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:30:00
#SBATCH -o logs/hf_check_%j.out
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
set -u
M=$(ls "$DEST"/.fetch/manifest_*.tsv | head -1)
missing=0
have=0
while IFS=$'\t' read -r path size sha; do
  if [ -f "$DEST/$path" ] && [ "$(stat -c %s "$DEST/$path")" = "$size" ]; then
    have=$((have + 1))
  else
    echo "missing or incomplete: $path"
    missing=$((missing + 1))
  fi
done < "$M"
echo "$REPO in $DEST: $have files complete, $missing missing"
if [ $missing = 0 ]; then
  echo "COMPLETE (weights checked against sha256 during download)"
else
  echo "rerun slurm/fetch_hf_repo.sh with the same arguments to fetch the rest"
fi
