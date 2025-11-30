import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import autotag
import capping
import select_caps
from QuickRename import rename_files_in_hierarchy


# Bundle-local paths (portable, descriptive)
BUNDLE_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = BUNDLE_ROOT / "input_videos"            # place source videos/folders here
FRAMES_ROOT = BUNDLE_ROOT / "frames_capped"          # ffmpeg output
WORKSPACE_ROOT = BUNDLE_ROOT / "workspace"           # selection + crops live here
WORKSPACE_RAW = WORKSPACE_ROOT / "raw"               # capped frames before selection
ARCHIVE_MP4 = BUNDLE_ROOT / "archive_mp4"            # MP4 archive (flat)
FINAL_OUTPUT = BUNDLE_ROOT / "final_ready"           # optional final move target

WORKING_SELECTED = WORKSPACE_ROOT  # selected images live directly in WORKSPACE_ROOT

# Workflow switches
ENABLE_FINAL_MOVE = True  # move selected+cropped output to FINAL_OUTPUT at the end

# Scripts
CROP_SCRIPT = Path(__file__).resolve().parent / "crop_and_flip.Bulk.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset workflow bundle")
    parser.add_argument("--autotag", action="store_true", help="Run wd-eva02 tagging on CPU after final move")
    parser.add_argument("--autotag-config", type=Path, help="Path to autotag.config.json (optional override)")
    parser.add_argument("--autotag-threshold", type=float, help="Override general tag threshold")
    parser.add_argument("--autotag-character-threshold", type=float, help="Override character tag threshold")
    parser.add_argument("--autotag-max-tags", type=int, help="Limit number of tags per image (including trigger)")
    parser.add_argument("--autotag-model", help="Override the tagger model id")
    return parser.parse_args()


def log(msg: str) -> None:
    print(f"[info] {msg}")


def ensure_dir_empty(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def move_videos_flat(paths: Iterable[Path], src_root: Path, dst_root: Path) -> List[Path]:
    """
    Move only files (e.g., MP4) into dst_root without recreating source folders.
    Names are prefixed with the relative path to avoid collisions.
    """
    moved: List[Path] = []
    dst_root.mkdir(parents=True, exist_ok=True)
    for src in paths:
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        name_parts = list(rel.parts)
        base_name = "_".join(name_parts)
        dst = dst_root / base_name
        # ensure unique
        if dst.exists():
            stem, suffix = dst.stem, dst.suffix
            counter = 1
            while dst.exists():
                dst = dst_root / f"{stem}_{counter}{suffix}"
                counter += 1
        log(f"Move {src} -> {dst}")
        shutil.move(str(src), dst)
        moved.append(dst)
    return moved


def move_capping_to_raw() -> List[Path]:
    moved: List[Path] = []
    if not FRAMES_ROOT.exists():
        return moved
    for entry in sorted(FRAMES_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        rel = entry.name
        dst = WORKSPACE_RAW / rel
        if dst.exists():
            shutil.rmtree(dst)
        log(f"Move capped frames {entry} -> {dst}")
        shutil.move(str(entry), dst)
        moved.append(dst)
    # Clean up empty folders left behind
    for entry in sorted(FRAMES_ROOT.glob("**/*"), reverse=True):
        if entry.is_dir():
            try:
                entry.rmdir()
            except OSError:
                pass
    return moved


def move_source_dirs_to_working() -> List[Path]:
    """
    Move remaining folders from INPUT_ROOT into WORKSPACE_ROOT (after MP4 removal).
    If the target exists, merge contents and ensure unique filenames.
    """
    moved: List[Path] = []
    if not INPUT_ROOT.exists():
        return moved

    def move_item(child: Path, dest_dir: Path) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / child.name
        if target.exists():
            stem, suffix = target.stem, target.suffix
            counter = 1
            while target.exists():
                target = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        log(f"Move {child} -> {target}")
        shutil.move(str(child), target)

    for folder in sorted(INPUT_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        dest = WORKSPACE_ROOT / folder.name
        if dest.exists():
            for child in sorted(folder.iterdir()):
                move_item(child, dest)
            try:
                folder.rmdir()
            except OSError:
                pass
        else:
            log(f"Move {folder} -> {dest}")
            shutil.move(str(folder), dest)
        moved.append(dest)
    return moved


def run_crop_and_flip(target_dirs: Iterable[Path]) -> None:
    for folder in target_dirs:
        if not folder.is_dir():
            continue
        log(f"Crop+Flip in {folder}")
        subprocess.run(
            [sys.executable, str(CROP_SCRIPT), str(folder)],
            check=True,
        )


def main(args: argparse.Namespace) -> None:
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Source root not found: {INPUT_ROOT}")
    FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
    ARCHIVE_MP4.mkdir(parents=True, exist_ok=True)
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

    log("Step 1: QuickRename in input_videos")
    rename_files_in_hierarchy(INPUT_ROOT)

    log("Prepare raw folder in Working")
    ensure_dir_empty(WORKSPACE_RAW)
    old_selected = WORKSPACE_ROOT / "selected"
    if old_selected.exists():
        shutil.rmtree(old_selected)

    log("Step 2: Capping videos (12 fps)")
    produced = capping.cap_all(INPUT_ROOT, FRAMES_ROOT)
    log(f"Capping complete: {len(produced)} video folders")

    log("Step 2.5: Archive MP4s to archive_mp4 (files only, flat)")
    videos = list(capping.iter_videos(INPUT_ROOT))
    move_videos_flat(videos, INPUT_ROOT, ARCHIVE_MP4)

    log("Step 2.5: Move capped frames into workspace/raw")
    move_capping_to_raw()

    log("Step 2.5: Move remaining source folders into workspace (merge)")
    move_source_dirs_to_working()

    log("Step 3: Select caps (40 per character, 10 face quota)")
    select_caps.run_selection(
        src_root=WORKSPACE_RAW,
        dest_root=WORKING_SELECTED,
    )

    log("Step 4: Crop and Flip selected images")
    target_dirs = [
        p for p in WORKING_SELECTED.iterdir()
        if p.is_dir() and not p.name.startswith("_") and p.name != "selected"
    ]
    run_crop_and_flip(target_dirs)

    if ENABLE_FINAL_MOVE:
        log("Step 5: Move finished set to final_ready")
        FINAL_OUTPUT.mkdir(parents=True, exist_ok=True)
        for folder in target_dirs:
            dst = FINAL_OUTPUT / folder.name
            if dst.exists():
                raise RuntimeError(f"Destination already exists, refusing to overwrite: {dst}")
            log(f"Move {folder} -> {dst}")
            shutil.move(str(folder), dst)

    if args.autotag:
        tag_root = FINAL_OUTPUT if ENABLE_FINAL_MOVE else WORKING_SELECTED
        log(f"Step 6: Autotag images in {tag_root}")
        autotag.tag_folder(
            root=tag_root,
            model_id=args.autotag_model,
            general_threshold=args.autotag_threshold,
            character_threshold=args.autotag_character_threshold,
            max_tags=args.autotag_max_tags,
            config_path=args.autotag_config,
        )

    log("Workflow done.")


if __name__ == "__main__":
    main(parse_args())
