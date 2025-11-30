# FrameForge (dataset workflow bundle)

FrameForge automates creation of character training datasets: rename source clips, cap frames, pick diverse samples, crop/flip, and optionally stage the final set. Place character videos in input_videos, grouped by character folder, and include a representative full-body cover JPG inside each folder. The script does not remove background actors or other artifacts — it's best used with clean inputs (e.g., Grok-generated videos; works best with Grok Imagine: grok.com/imagine/).

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
