import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms

DEFAULT_MODEL_ID = "SmilingWolf/wd-eva02-large-tagger-v3"
DEFAULT_GENERAL_THRESHOLD = 0.4
DEFAULT_CHARACTER_THRESHOLD = 0.4
DEFAULT_MAX_TAGS = 30
CONFIG_FILE = Path(__file__).resolve().parent / "autotag.config.json"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
TAG_FILE = "selected_tags.csv"
WEIGHTS_FILE = "model.safetensors"
CONFIG_JSON = "config.json"
CHARACTER_CATEGORY = 4
RATING_CATEGORY = 9


@dataclass
class TaggerSettings:
    model_id: str = DEFAULT_MODEL_ID
    general_threshold: float = DEFAULT_GENERAL_THRESHOLD
    character_threshold: float = DEFAULT_CHARACTER_THRESHOLD
    max_tags: int = DEFAULT_MAX_TAGS


def iter_images(root: Path) -> Iterable[Path]:
    for suffix in IMAGE_SUFFIXES:
        yield from root.rglob(f"*{suffix}")


def load_config(path: Optional[Path] = None) -> TaggerSettings:
    """
    Load tagger settings from JSON file. Missing keys fall back to defaults.
    """
    config_path = path or CONFIG_FILE
    settings = TaggerSettings()
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        settings.model_id = data.get("model_id", settings.model_id)
        settings.general_threshold = float(data.get("general_threshold", settings.general_threshold))
        settings.character_threshold = float(data.get("character_threshold", settings.character_threshold))
        settings.max_tags = int(data.get("max_tags", settings.max_tags))
    else:
        # create a template for the user to edit next time
        config_path.write_text(
            json.dumps(
                {
                    "model_id": settings.model_id,
                    "general_threshold": settings.general_threshold,
                    "character_threshold": settings.character_threshold,
                    "max_tags": settings.max_tags,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return settings


@lru_cache(maxsize=2)
def load_labels_and_model(
    model_id: str,
) -> Tuple[List[str], List[int], torch.nn.Module, transforms.Compose]:
    """
    Download model + labels from HF, build timm model on CPU, and return inference transform.
    Cached per model_id to avoid reload.
    """
    tag_path = hf_hub_download(repo_id=model_id, filename=TAG_FILE)
    cfg_path = hf_hub_download(repo_id=model_id, filename=CONFIG_JSON)
    weights_path = hf_hub_download(repo_id=model_id, filename=WEIGHTS_FILE)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Labels / categories
    names: List[str] = []
    categories: List[int] = []
    with open(tag_path, "r", encoding="utf-8") as f:
        # skip header
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            _, name, category, *_ = parts
            names.append(name)
            categories.append(int(category))

    arch = cfg.get("architecture", "eva02_large_patch14_448")
    num_classes = len(names)
    model = timm.create_model(arch, num_classes=num_classes, pretrained=False)
    state = load_safetensors(weights_path)
    model.load_state_dict(state, strict=True)
    model.to("cpu")
    model.eval()

    data_cfg = resolve_data_config(model=model, use_test_size=False)
    # override with provided mean/std/size if present
    pretrained_cfg = cfg.get("pretrained_cfg", {})
    mean = pretrained_cfg.get("mean", data_cfg["mean"])
    std = pretrained_cfg.get("std", data_cfg["std"])
    input_size = pretrained_cfg.get("input_size", data_cfg["input_size"])
    transform = create_transform(
        input_size=tuple(input_size),
        mean=mean,
        std=std,
        interpolation=pretrained_cfg.get("interpolation", "bicubic"),
        crop_pct=pretrained_cfg.get("crop_pct", 1.0),
        crop_mode=pretrained_cfg.get("crop_mode", "center"),
    )

    return names, categories, model, transform


def predict_tags(
    image_path: Path,
    labels: List[str],
    categories: List[int],
    model: torch.nn.Module,
    transform: transforms.Compose,
    general_threshold: float,
    character_threshold: float,
    max_tags: int,
    trigger_tag: Optional[str] = None,
) -> str:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.inference_mode():
        logits = model(tensor)
    probs = torch.sigmoid(logits)[0].tolist()

    tags: List[Tuple[float, str]] = []
    for prob, name, category in zip(probs, labels, categories):
        if category == RATING_CATEGORY:
            continue  # skip ratings
        clean_name = name.replace("_", " ")
        if category == CHARACTER_CATEGORY:
            if prob >= character_threshold:
                tags.append((prob, clean_name))
        elif prob >= general_threshold:
            tags.append((prob, clean_name))

    tags.sort(key=lambda t: t[0], reverse=True)

    parts: List[str] = []
    if trigger_tag:
        parts.append(trigger_tag)

    remaining_slots = max(max_tags - len(parts), 0)
    parts.extend(tag for _, tag in tags[:remaining_slots])

    return ", ".join(parts)


def tag_folder(
    root: Path,
    model_id: Optional[str] = None,
    general_threshold: Optional[float] = None,
    character_threshold: Optional[float] = None,
    max_tags: Optional[int] = None,
    config_path: Optional[Path] = None,
) -> None:
    if not root.exists():
        raise FileNotFoundError(f"Tag target not found: {root}")
    settings = load_config(config_path)
    if model_id is not None:
        settings.model_id = model_id
    if general_threshold is not None:
        settings.general_threshold = general_threshold
    if character_threshold is not None:
        settings.character_threshold = character_threshold
    if max_tags is not None:
        settings.max_tags = max_tags

    labels, categories, model, transform = load_labels_and_model(settings.model_id)
    for image_path in iter_images(root):
        trigger = image_path.parent.name
        tag_line = predict_tags(
            image_path=image_path,
            labels=labels,
            categories=categories,
            model=model,
            transform=transform,
            general_threshold=settings.general_threshold,
            character_threshold=settings.character_threshold,
            max_tags=settings.max_tags,
            trigger_tag=trigger,
        )
        out_path = image_path.with_suffix(".txt")
        out_path.write_text(tag_line, encoding="utf-8")
