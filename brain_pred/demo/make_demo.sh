#!/bin/bash
# Replay + render one recorded play into a demo video (CPU; env with stable-retro, nilearn, PyAV: $MJG_PY).
#   bash brain_pred/demo/make_demo.sh <play dir> <out.mp4> [subject] [encoder model] [render.py options...]
#   e.g. ... sub-03 <encoder name> --highlight top-k --top-k 200   (colour only the 200 best-predicted regions)
# The encoder is $MJG_ROOT/results/bold_generator/<subject>/<encoder model>.npz (encode.py --save).
# Run on a compute node, e.g. srun -p short -c 4 --mem=16G -t 00:45:00 bash brain_pred/demo/make_demo.sh ...
set -euo pipefail
source slurm/env.sh
PLAY=$1
OUT=$2
SUB=${3:-sub-03}
MODEL=${4:-gamestate}
mkdir -p "$MJG_SCRATCH/tmp/demo"
RP=$MJG_SCRATCH/tmp/demo/$(echo $PLAY | tr / _).replay_ll.npz  # with the low-level confounds
if [ ! -f $RP ]; then
  "$MJG_PY" -u brain_pred/generator/replay.py --play $PLAY --out $RP
fi
HID=""
if [ -f $PLAY/hidden.npy ]; then
  HID="--hidden $PLAY/hidden.npy"
fi
"$MJG_PY" -u brain_pred/demo/render.py --play $PLAY --replay $RP \
    --model $MJG_ROOT/results/bold_generator/$SUB/$MODEL.npz --subject $SUB $HID --out $OUT "${@:5}"
