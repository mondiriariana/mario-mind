#!/bin/bash
# One job of slurm/fetch_hf_repo.sh: takes every N-th file of the repository (by task id), downloads
# it with resume, checks size and sha256, and moves it into place. Already complete files are
# skipped, so a rerun picks up where an interrupted one stopped. Reads REPO, DEST and N from the environment.
#SBATCH -J hf-repo
#SBATCH -p medium
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/hf_repo_%A_%a.out
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
set -u
AUTH=()
if [ -s "$HF_HOME/token" ]; then
  AUTH=(-H "Authorization: Bearer $(cat "$HF_HOME/token")")
fi
U=https://huggingface.co/$REPO/resolve/main
code=$(curl -sL -o /dev/null -w "%{http_code}" "${AUTH[@]}" "$U/config.json")
if [ "$code" != "200" ]; then
  echo "access check failed for $REPO (HTTP $code): gated? token missing or access not granted yet"
  exit 1
fi
M="$DEST/.fetch/manifest_$SLURM_ARRAY_TASK_ID.tsv"
curl -sf --retry 5 "${AUTH[@]}" "https://huggingface.co/api/models/$REPO/tree/main?recursive=1" | "$MJG_PY" -c '
import json, sys
for f in sorted(json.load(sys.stdin), key=lambda f: f["path"]):
    if f["type"] == "file":
        print(f["path"], f["size"], (f.get("lfs") or {}).get("oid", "-"), sep="\t")' > "$M"
echo "$(date) $REPO: $(wc -l < "$M") files, $(awk -F'\t' '{s+=$2} END {printf "%.1f GB", s/1e9}' "$M") on $(hostname)"
i=0
failed=0
while IFS=$'\t' read -r path size sha; do
  i=$((i + 1))
  if [ $(( (i - 1) % N )) != "$SLURM_ARRAY_TASK_ID" ]; then
    continue
  fi
  out="$DEST/$path"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ]; then
    continue
  fi
  for try in $(seq 1 20); do
    if curl -sfL --retry 3 "${AUTH[@]}" -C - -o "$out.part" "$U/$path" \
        && [ "$(stat -c %s "$out.part" 2>/dev/null)" = "$size" ]; then
      break
    fi
    got=$(stat -c %s "$out.part" 2>/dev/null || echo 0)
    echo "$(date) $path: attempt $try stopped at $got of $size bytes, resuming"
    sleep 30
  done
  if [ "$(stat -c %s "$out.part" 2>/dev/null)" != "$size" ]; then
    echo "FAILED $path"
    failed=$((failed + 1))
    continue
  fi
  if [ "$sha" != "-" ] && [ "$(sha256sum "$out.part" | cut -d' ' -f1)" != "$sha" ]; then
    echo "CHECKSUM MISMATCH $path (removed, rerun to fetch again)"
    rm -f "$out.part"
    failed=$((failed + 1))
    continue
  fi
  mv "$out.part" "$out"
  echo "$(date) done $path ($size bytes)"
done < "$M"
echo "$(date) job $SLURM_ARRAY_TASK_ID finished, $failed failed"
[ $failed = 0 ]
