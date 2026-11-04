#!/bin/bash
# One task of slurm/fetch_hf_model.sh. Task 0: access check, file list, small files. Tasks 1..N: one byte range
# each of $BIG, resumed across attempts. Reads REPO, DEST, BIG and N from the environment.
#SBATCH -J hf-get
#SBATCH -p medium
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 12:00:00
#SBATCH -o logs/hf_get_%A_%a.out
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
set -u
AUTH=()
if [ -s "$HF_HOME/token" ]; then
  AUTH=(-H "Authorization: Bearer $(cat "$HF_HOME/token")")
fi
U=https://huggingface.co/$REPO/resolve/main
code=$(curl -sL -o /dev/null -w "%{http_code}" "${AUTH[@]}" "$U/config.json")  # -L: resolve URLs redirect
if [ "$code" != "200" ]; then
  echo "access check failed for $REPO (HTTP $code): gated? token missing or access not granted"
  exit 1
fi
TREE="$DEST/parts/tree.json"
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  curl -sf "${AUTH[@]}" "https://huggingface.co/api/models/$REPO/tree/main" > "$TREE" && echo "got file list"
  small=$("$MJG_PY" -c '
import json, sys
names = []
for x in json.load(open(sys.argv[1])):
    if x["type"] == "file" and x["size"] < 50_000_000:
        names.append(x["path"])
print(" ".join(names))
' "$TREE")
  for f in $small; do
    mkdir -p "$DEST/$(dirname "$f")"
    curl -sfL --retry 5 "${AUTH[@]}" -o "$DEST/$f" "$U/$f" && echo "got $f"
  done
  exit 0
fi
SIZE=$(curl -sf "${AUTH[@]}" "https://huggingface.co/api/models/$REPO/tree/main" | "$MJG_PY" -c '
import json, sys
for f in json.load(sys.stdin):
    if f["path"] == sys.argv[1]:
        print(f["size"])
        break
else:
    sys.exit(sys.argv[1] + " is not in the file list")
' "$BIG")
i=$SLURM_ARRAY_TASK_ID
chunk=$(( (SIZE + N - 1) / N ))
s=$(( (i - 1) * chunk ))
e=$(( i * chunk - 1 ))
if [ $e -ge $SIZE ]; then
  e=$(( SIZE - 1 ))
fi
want=$(( e - s + 1 ))
out="$DEST/parts/part_$(printf %02d $i)"
echo "$(date) $REPO piece $i/$N: bytes $s-$e of $SIZE on $(hostname)"
for try in $(seq 1 12); do
  have=$(stat -c %s "$out.tmp" 2>/dev/null || echo 0)  # resume: fetch only the bytes still missing
  if [ "$have" = "$want" ]; then
    break
  fi
  if [ "$have" -gt "$want" ]; then
    rm -f "$out.tmp"
    have=0
  fi
  if curl -sfL "${AUTH[@]}" -r $(( s + have ))-$e "$U/$BIG" >> "$out.tmp" \
      && [ "$(stat -c %s "$out.tmp")" = "$want" ]; then
    break
  fi
  echo "$(date) attempt $try stopped at $(stat -c %s "$out.tmp" 2>/dev/null || echo 0) of $want bytes, resuming"
  sleep 30
done
if [ "$(stat -c %s "$out.tmp")" != "$want" ]; then
  echo "SIZE MISMATCH"
  exit 1
fi
mv "$out.tmp" "$out"
echo "$(date) done piece $i"
