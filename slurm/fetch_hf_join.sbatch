#!/bin/bash
# Final job of slurm/fetch_hf_model.sh: join the pieces of $BIG and check the sha256 Hugging Face lists for it.
# Reads REPO, DEST and BIG from the environment (set by fetch_hf_model.sh).
#SBATCH -J hf-join
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH -o logs/hf_join_%j.out
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
set -eu
cat "$DEST"/parts/part_?? > "$DEST/$BIG.tmp"
want=$("$MJG_PY" -c '
import json, sys
for f in json.load(open(sys.argv[1])):
    if f["path"] == sys.argv[2]:
        print(f["lfs"]["oid"])
        break
else:
    sys.exit(sys.argv[2] + " is not in the file list")
' "$DEST/parts/tree.json" "$BIG")
got=$(sha256sum "$DEST/$BIG.tmp" | cut -d" " -f1)
echo "expected $want"
echo "got      $got"
if [ "$want" != "$got" ]; then
  echo "CHECKSUM MISMATCH"
  exit 1
fi
mv "$DEST/$BIG.tmp" "$DEST/$BIG"
rm -f "$DEST"/parts/part_??
echo "$REPO ready in $DEST"
