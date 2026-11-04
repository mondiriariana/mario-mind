# Data

## Source

CNeuroMod `mario` (Paugam et al., 2025, bioRxiv 10.1101/2025.11.28.691119), `algonauts2027` branch
of the public datalad repository (commit b778ff6). Each *play* (repetition) is one attempt sequence at
one level, stored as

- `sub-XX_ses-YYY_task-mario_level-wXlY_rep-NNN.bk2`: stable-retro replay (zip): start savestate
  `Core.bin` plus the controller log (`Input Log.txt`, one line per 60 Hz frame);
- `..._recording.mp4`: 240×224 h264 video at 60 fps, with audio;
- `mario.annotations/.../*_desc-annotated_events.tsv`: per-run events (button intervals, kills,
  power-ups, deaths, level boundaries) with emulator frame indices.

fMRI comes from `mario.fmriprep`; `mjg/encoding.py` reads its confounds and the parcel time series derived
from it (Schaefer-1000).

The game integration (`SuperMarioBros-Nes`: ROM, `data.json` RAM map, 32 level states) is from
`courtois-neuromod/mario.stimuli` (commit e3ecccf), whose annexed files come from CNeuroMod's public CONP
mirror. ROM sha1 (without the 16-byte iNES header) = `facee9c577a5262dbe33ac4930bb0b58c8c037f7`, as listed in
the integration's `rom.sha`. The ROM is never stored in this repository.

## Subjects and held-out sessions

Five subjects (sub-01, 02, 03, 05, 06), 21 to 29 sessions and 602 to 797 plays each, 22 levels.
`configs/holdout_sessions.json` fixes the sessions held out for the fMRI analysis: per subject every 5th
session (by name), plus one added session when a level was missing from the held-out set, about 20% of each
subject's plays, with all 22 levels in every held-out set. These sessions are never used to write reasoning
traces or to train or validate the agent; every script that splits data reads this file.

## Frame alignment

Checked on `sub-01_ses-001_..._w1l1_rep-000`:

- **Replay screen k = video frame k.** The mean absolute pixel error between replay and video is lowest at
  offset 0 with no crop (6.5 grey levels, the h264 compression; offsets −1 / +1 give 7.8 / 9.8).
  stable-retro's screen is already the 224×240 overscan-cropped image the videos show.
- **Inputs.** The input log has one leading line more than the replay: log line k+1 = the input applied
  on replay frame k (100% agreement on all 8 buttons at that offset).
- **Annotations.** Every button interval's `frame_start` is one frame before the replay's first pressed
  frame, i.e. annotation frame indices count log lines.

Screens, inputs and RAM are therefore all taken from the emulator replay itself, which keeps human and agent
observations identical in format: the agent's screens come from the same emulator.

## Steps, clips, actions

- Decision step t starts at frame f = 4t (15 Hz). Observation = 8 screens f−15, f−13, …, f−1 (`mjg/data.py`).
- Action label = the human's buttons on frame f, projected onto 14 combos (NOOP; RIGHT/LEFT with
  any of A, B; A; B; A+B; DOWN; UP) by `mjg.actions.buttons_to_action`, which flags the frames it cannot
  represent exactly (e.g. LEFT+RIGHT together, or UP/DOWN with a direction).
- Controllable steps: RAM `player_state ∈ {8, 9, 10, 12}` and `oper_mode = 1`; the others are level intros,
  death animations, pipe transitions and flagpole sequences.

## Derived features (`$MJG_ROOT/features/`, built by `scripts/`)

| Folder | Content | Built by |
|---|---|---|
| `vjepa21G_c16s8_f8_g8/` | V-JEPA 2.1 ViT-G, one row every 8 screens: clip of 16 screens 8 frames apart, 4 depths (blocks 11, 23, 37, 47), 8×8 grid, D = 1664; plus the newest tubelet (`.last.npy`) and the frame indices, buttons and RAM variables (`.game.npz`) | `scripts/features/encode_frames.py` (`slurm/encode_vjepa21G_full.sbatch`) |
| `vjepa21G_c16s8_f8_compact/` | the same rows as grid means and 2×2 quadrants | `scripts/features/compact_vjepa21G.py` |
| `vjepa21G_c16s8_f8_g8_rowmask/` | `masks.npz` (gameplay / in_control per row) and `summary.json` | `scripts/features/build_row_masks.py` |
| `ram_full/` | the full 2 KB NES RAM and the buttons of every frame of every play | `scripts/data/extract_full_ram.py` |

`$MJG_ROOT/results/moment_index/v1/` holds the moment index (`moments.csv`, `coverage.json`) VLA-JEPA's frame
set is drawn from (`scripts/data/build_moment_index.py`).

## V-JEPA 2.1 G row masks

The feature folder covers 3,374 plays, but 19% of its rows are not gameplay: after a game over the recordings
continue through the game-over screen, the title screen and the attract demo (1,411 plays; none of them
resumes play afterwards). The feature files are left as they are; the masks say which rows to use:

| mask | rule | rows kept |
|---|---|---|
| `gameplay` | whole 2 s clip in play mode (RAM 0x770 == 1) | 1,836,456 (81.0%) |
| `in_control` | gameplay + Mario in normal control (0x0E == 8) + game timer running (moment-index rule) | 1,342,102 (59.2%) |

Load with `mjg.data.load_vjepa_play(feat_dir, play_id, rows="gameplay" | "in_control" | "all", depth=None)`,
which returns the kept rows, their row indices and their screens. The default (`gameplay`) skips the
non-gameplay rows. Use `in_control` for training on the person's play; use `all` only for the fMRI encoding
(the participant saw those screens; add a regressor for them).

## Not used

- The mp4 audio track (sound features for the nuisance model are computed from the emulator replays instead).
- The `_variables.json` / `_lowlevel.npy` files, which are not public. Symbolic state comes from emulator RAM
  (`data.json` map).
- Eye tracking: collected but not released for Mario. Physiology (ECG, pulse, respiration, skin conductance
  per run) is in `mario`, with a preprocessed version in `courtois-neuromod/mario.physprep`; neither is used.
