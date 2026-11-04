# SFT settings, sourced by job_sft.sbatch: sets MODEL and ARGS (the train_sft.py flags).
# Required:
#   SFT_MODEL            the VLA-JEPA Glimmer model folder (experiments/glimmer_vlajepa output)
# Optional:
#   SFT_TAG              run name (default sft); output in $MJG_ROOT/results/glimmer_imitation/sft/<tag>
#   SFT_DATA             build_sft.py output, one sub-NN folder per subject (default $MJG_SCRATCH/glimmer_sft/all)
#   SFT_HUMAN_REASONING  gen_human_reasoning.py output for those turns
#   SFT_TRACES           audited Odysseus traces (experiments/odysseus/audit_dataset.py output)
# Recipe: a LoRA on the VLA-JEPA Glimmer, 5 subjects with the player tag; every batch = 11 human turns [the model's
# own reasoning (it never sees the button), no loss] -> the human's button (loss) + 5 Odysseus traces with the
# screenshots of up to 8 earlier turns (loss on perception + reasoning); 354 steps.
MODEL=${SFT_MODEL:?path to the VLA-JEPA Glimmer model folder}
D=${SFT_DATA:-$MJG_SCRATCH/glimmer_sft/all}
HR=${SFT_HUMAN_REASONING:-$MJG_ROOT/results/glimmer_imitation/human_reasoning_all5_v1.jsonl}
TRACES=${SFT_TRACES:-$MJG_ROOT/results/odysseus_sft/v1_audited}
if [ ! -s "$HR" ]; then
  echo "$HR missing: write it with gen_human_reasoning.py (job_gen_reasoning.sbatch)"
  exit 1
fi
ARGS=(--data-dirs "$D/sub-01" "$D/sub-02" "$D/sub-03" "$D/sub-05" "$D/sub-06" --max-plays 300 --max-val-plays 40
      --human-reasoning "$HR"
      --traces "$TRACES" --trace-frac 0.3125
      --trace-history 8 --steps 354 --bs 16 --evals 6 --tag "${SFT_TAG:-sft}")
