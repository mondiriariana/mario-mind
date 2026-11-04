# scripts/

Data steps the pipeline builds on, run once over every human play. Run from the repo root, e.g.
`python scripts/features/encode_frames.py ...`, or on the cluster through the matching `slurm/*.sbatch`.

## features/
Frozen V-JEPA 2.1 G run over every human play; its features are what VLA-JEPA learns to predict and a brain-encoding
band.
- `encode_frames.py`: dense V-JEPA features, one clip every 8 screens of every play, at 4 depths
  (`slurm/encode_vjepa21G_full.sbatch`).
- `compact_vjepa21G.py`: compact copy of those features (the copy the later steps read).
- `build_row_masks.py`: which feature rows are gameplay and which have Mario in control.

## data/
- `extract_full_ram.py`: full NES RAM for every frame of every human play (game state, danger, brain features).
- `build_moment_index.py`: index of the moments VLA-JEPA's frame set is drawn from.

Order: `extract_full_ram.py` and `encode_frames.py` first (independent), then `build_row_masks.py`
(needs both for its timer check), `compact_vjepa21G.py` and `build_moment_index.py`.
