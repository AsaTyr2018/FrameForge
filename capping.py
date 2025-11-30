import subprocess
from pathlib import Path
from typing import Iterable, List

# Configuration for frame extraction
VIDEO_EXTS = {".mp4", ".mov", ".mkv"}
FPS = 12  # export 12 frames per second
JPEG_QUALITY = 2  # lower is better quality for ffmpeg qscale (2 is near-lossless)


def iter_videos(root: Path) -> Iterable[Path]:
    """Yield all video files under root that match VIDEO_EXTS."""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            yield path


def cap_video(src: Path, out_dir: Path) -> None:
    """
    Export frames from a single video to out_dir using ffmpeg.
    Skips work if out_dir already has files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        print(f"[skip] frames already exist: {out_dir}")
        return

    output_pattern = out_dir / "%06d.jpg"
    cmd = [
        "ffmpeg",
        "-loglevel",
        "warning",
        "-i",
        str(src),
        "-vf",
        f"fps={FPS}",
        "-qscale:v",
        str(JPEG_QUALITY),
        str(output_pattern),
    ]
    print(f"[cap ] {src} -> {out_dir}")
    subprocess.run(cmd, check=True)


def cap_all(
    source_root: Path,
    capping_root: Path,
) -> List[Path]:
    """
    Cap all videos under source_root into capping_root, mirroring the folder structure.
    Returns list of produced frame directories.
    """
    produced: List[Path] = []
    for video in sorted(iter_videos(source_root)):
        rel_parent = video.parent.relative_to(source_root)
        out_dir = capping_root / rel_parent / video.stem
        cap_video(video, out_dir)
        produced.append(out_dir)
    return produced


if __name__ == "__main__":
    raise SystemExit("Use this module from workflow.py")
