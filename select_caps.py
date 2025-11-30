from __future__ import annotations

"""
Selects up to 40 diverse images per character (e.g., MiniJupiter_1 + MiniJupiter_2)
and copies them into a consolidated output folder.
Runs stand-alone or is imported by workflow.py.
"""

import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image

# Configuration (bundle-local paths by default)
BUNDLE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = BUNDLE_ROOT / "ScreenCaps" / "03.Working" / "_raw"  # filled with capped frames
DEST_ROOT = BUNDLE_ROOT / "ScreenCaps" / "03.Working"  # final picks land here (per character)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_PER_CHARACTER = 40
FACE_QUOTA = 10  # target number of close-ups; rest will be non-face
HAMMING_THRESHOLD = 6  # increase for more variety, decrease if too few images selected
HAMMING_RELAXED = 4  # fallback threshold if diversity is too strict


def iter_images(folder: Path) -> Iterable[Path]:
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def character_key(folder_name: str) -> str:
    """
    Collapse MiniJupiter_1, MiniJupiter_2 -> MiniJupiter
    If no trailing numeric part exists, use the full name.
    """
    if "_" in folder_name:
        base, last = folder_name.rsplit("_", 1)
        if last.isdigit():
            return base
    return folder_name


def phash(path: Path, size: int = 8) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size * 4, size * 4), Image.LANCZOS)
    dct = np.fft.fft2(np.asarray(img, dtype=np.float32))
    dct_low = np.real(dct)[:size, :size]
    median = np.median(dct_low)
    bits = (dct_low > median).astype(np.uint8).flatten()
    return bits


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def select_diverse(
    paths: List[Path],
    limit: int,
    seed_hashes: List[np.ndarray] | None = None,
) -> List[Tuple[Path, np.ndarray]]:
    """
    Select images that are at least HAMMING_THRESHOLD apart.
    If not enough are found, relax to HAMMING_RELAXED.
    """
    selected: List[Tuple[Path, np.ndarray]] = []
    used_paths: set[Path] = set()

    def run_with_threshold(threshold: int, seeds: List[np.ndarray]) -> None:
        hashes: List[np.ndarray] = list(seeds)
        nonlocal selected, used_paths
        for p in sorted(paths):
            if p in used_paths:
                continue
            try:
                h = phash(p)
            except Exception:
                continue
            if all(hamming(h, prev_hash) >= threshold for prev_hash in hashes):
                hashes.append(h)
                selected.append((p, h))
                used_paths.add(p)
            if len(selected) >= limit:
                break

    seeds = list(seed_hashes) if seed_hashes else []
    run_with_threshold(HAMMING_THRESHOLD, seeds)
    if len(selected) < limit:
        run_with_threshold(HAMMING_RELAXED, seeds)
    return selected


def run_selection(
    src_root: Path = SRC_ROOT,
    dest_root: Path = DEST_ROOT,
    face_quota: int = FACE_QUOTA,
    target_per_char: int = TARGET_PER_CHARACTER,
) -> None:
    if not src_root.exists():
        print(f"[warn] source root not found: {src_root}")
        return
    dest_root.mkdir(parents=True, exist_ok=True)

    # Group folders by character key
    grouped = defaultdict(list)
    for folder in sorted(p for p in src_root.iterdir() if p.is_dir() and p.name != "00"):
        grouped[character_key(folder.name)].append(folder)

    for char, folders in grouped.items():
        all_images: List[Path] = []
        for folder in folders:
            all_images.extend(iter_images(folder))

        face_imgs = [p for p in all_images if any("face" in part.lower() for part in p.parts)]
        other_imgs = [p for p in all_images if p not in face_imgs]

        chosen_pairs: List[Tuple[Path, np.ndarray]] = []

        face_selected = select_diverse(face_imgs, limit=face_quota)
        chosen_pairs.extend(face_selected)

        face_hashes = [h for _, h in face_selected]
        remaining = target_per_char - len(face_selected)
        if remaining > 0:
            other_selected = select_diverse(other_imgs, limit=remaining, seed_hashes=face_hashes)
            chosen_pairs.extend(other_selected)

        # If still short, pull whatever is left (no diversity checks)
        if len(chosen_pairs) < target_per_char:
            missing = target_per_char - len(chosen_pairs)
            for p in sorted(other_imgs):
                if p in {c[0] for c in chosen_pairs}:
                    continue
                chosen_pairs.append((p, np.array([])))
                if len(chosen_pairs) >= target_per_char:
                    break

        chosen = [p for p, _ in chosen_pairs]
        if not chosen:
            print(f"[skip] {char}: no images found")
            continue

        out_dir = dest_root / char
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, src in enumerate(chosen, start=1):
            dest = out_dir / f"{idx}{src.suffix.lower()}"
            counter = 1
            while dest.exists():
                dest = out_dir / f"{idx}_{counter}{src.suffix.lower()}"
                counter += 1
            shutil.copy2(src, dest)
        print(f"[ok] {char}: copied {len(chosen)} images to {out_dir}")


def main() -> None:
    run_selection()


if __name__ == "__main__":
    main()
