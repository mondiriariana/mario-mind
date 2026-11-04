# Demo: the agent plays, reasons, and its predicted brain activity is shown in sync

Pipeline (the agent: VLA-JEPA Glimmer + its SFT adapter; the encoder: one saved by `encode.py --save`):
1. **Play**: `experiments/glimmer_imitation/glimmer_play.py --lora <SFT lora.pt> --save-hidden ...` records the play
   (per-frame inputs, the reply of every turn, the internal state per turn: last prompt token at 4 depths).
2. **Replay**: `brain_pred/generator/replay.py` rebuilds every frame and the per-frame RAM from the inputs (and checks
   that the final RAM equals the recorded one).
3. **BOLD**: the play's per-TR features (same code as for the human plays, `brain_pred/features`) go through a fitted
   encoder (`brain_pred/prediction`): predicted Schaefer-1000 BOLD per TR for one subject. Encoders live in
   `$MJG_ROOT/results/bold_generator/<sub>/<model>.npz` with their held-out-session accuracy (r per parcel).
4. **Render**: `render.py` writes the video: game | reasoning of the current turn | predicted brain on the surface
   (one image per TR) | time courses of 4 networks | internal-state trajectory.

```bash
bash brain_pred/demo/make_demo.sh <play dir> demo.mp4 sub-03 <encoder name> [--highlight top-k --top-k 200]
```

Limits, to be stated with the demo:
- The BOLD is a model prediction, not a measurement; its quality is the encoder's held-out accuracy, printed on the
  frame. Encoders with a V-JEPA band need per-TR V-JEPA features of the agent's own frames
  (`generate_bold.py --vjepa`); `render.py` takes encoders without one.
- The agent was trained on human play from all 22 levels: the demo play is new, the level is not.
- Game screens stay out of git and out of anything published (ROM / copyright); the video is for local viewing.
