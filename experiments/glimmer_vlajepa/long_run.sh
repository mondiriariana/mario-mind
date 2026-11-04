# Settings of the VLA-JEPA run, read by job_long.sbatch. Full fine-tune of Glimmer on the VLA-JEPA objective plus a
# next-token loss on Glimmer's own reasoning texts (gen_human_reasoning.py), so that it keeps its language.
# Up to 100 epochs with one validation per epoch. LR schedule wsd: warmup -> constant LR -> (1-sqrt) cooldown; the
# final cooldown (20% of the steps so far) starts after 5 validations without a new best, and the model after it is
# exported to $MJG_ROOT/results/glimmer_vlajepa/<TAG>/glimmer. --lm-balance 1 reweights the language loss every step
# so that its gradient on Glimmer's weights pulls as hard as the VLA-JEPA one.
# Optional: EXPORT_AT=YYYY-MM-DDTHH:MM also exports a milestone model at that time (after a 10% cooldown branch, to
# glimmer_step<N>/), so SFT and RL can start from it while training goes on.
LM=${LM:-$MJG_ROOT/results/glimmer_imitation/human_reasoning_all5_v1.jsonl}
TAG=full_h200_lm
ARGS=(--mode full --epochs 100 --evals 100 --patience 5 --bs 16 --mb 2 --warmup-steps 255
      --schedule wsd --cooldown-frac 0.2 --milestone-cooldown-frac 0.1
      --lm-data "$LM" --lm-balance 1 --lm-every 1
      --ckpt-every-min 120 --tag $TAG)
if [ -n "${EXPORT_AT:-}" ]; then
  ARGS+=(--export-at "$EXPORT_AT")
fi
# RESUME=1: continue from the latest checkpoint of this tag (after a job stopped at its time limit)
RESUME=${RESUME:-0}
