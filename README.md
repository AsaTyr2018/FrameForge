# FrameForge: what it is and does

FrameForge is a one-button pipeline for building character image datasets: drop your character folders (MP4s + a cover JPG) into `input_videos`, run `python workflow.py`, and it renames files, exports frames, picks a balanced subset, crops/flips them, and optionally autotags the final images. Clean inputs work best because the script does not remove background actors or artifacts. Outputs land in `final_ready` if you keep the final move on, otherwise in `workspace`.

Example use case: you have five short MP4 clips of a character. Put them in `input_videos/CharacterA` with a cover JPG, run `python workflow.py --autotag`, and you get a curated, cropped, tagged set of images in `final_ready/CharacterA` without manual frame picking.

## Structure (bundle-rooted)
- Folders: `input_videos` (place source videos/folders), `frames_capped` (ffmpeg frame output), `workspace/raw` (staging for selection), `workspace` (selected + cropped), `final_ready` (optional final drop), `archive_mp4` (flat MP4 archive)
- Scripts: `workflow.py` (orchestrator), `capping.py` (ffmpeg export), `select_caps.py` (diverse selector), `crop_and_flip.Bulk.py` (crop/flip helper), `QuickRename.py` (renamer)
- Other: `requirements.txt` dependencies

## Setup
1. Python 3.10+ recommended.
2. Install deps: `pip install -r requirements.txt`.
3. Ensure `ffmpeg` is installed and on PATH.
4. Paths are bundle-local by default; adjust the constants in `workflow.py` (and `select_caps.py` if run standalone) only if you want a different layout.

## Workflow (automated by `workflow.py`)
1) Rename files in `input_videos` by parent folder (`QuickRename.py`).
2) Cap all videos to JPEG frames in `frames_capped` at 12 fps (`capping.py` / `ffmpeg`).
3) Archive MP4s into `archive_mp4` (flat), move capped frames into `workspace/raw`, merge remaining sources into `workspace`.
4) Select up to 40 diverse images per character (10 face quota) into `workspace` (`select_caps.py`).
5) Crop and flip selected images in-place (`crop_and_flip.Bulk.py`).
6) If `ENABLE_FINAL_MOVE` is True, move the final folders into `final_ready`.
7) Optional: tag the finished set with `--autotag` (defaults in `autotag.config.json`; writes into `final_ready` if moved, otherwise `workspace`).

## Run
```
python workflow.py
# optional: add --autotag to run CPU tagging with SmilingWolf/wd-eva02-large-tagger-v3
# autotag settings live in autotag.config.json (thresholds, max tags, model id)
# optional overrides: --autotag-threshold --autotag-character-threshold --autotag-max-tags --autotag-model --autotag-config
python workflow.py --autotag
```

## Notes
- `crop_and_flip.Bulk.py` processes files named `1.jpg`, `2.png`, etc.; adjust or wrap if you need different patterns.
- Autotag: folder name is added as first tag, ratings are skipped, defaults use thresholds 0.4/0.4 and max 30 tags (edit in `autotag.config.json`).
