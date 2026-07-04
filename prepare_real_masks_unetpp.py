#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare manually corrected talc masks for U-Net++ training.

The exported manual masks can have CVAT/Label-Studio task names that do not
match source images. This script matches them by sorted order, combines multiple
mask PNGs for the same task id, creates visual debug overlays, and writes an
aligned dataset with identical image/mask basenames.

Example:
  python prepare_real_masks_unetpp.py \
    --clean_dir "path/to/clean_images" \
    --real_masks_dir "talc_masks_out/real_masks" \
    --out_dir "talc_unetpp_dataset"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
TASK_RE = re.compile(r"task-(\d+)-", re.IGNORECASE)


@dataclass
class PreparedItem:
    index: int
    split: str
    clean_path: str
    source_mask_paths: str
    image_path: str
    mask_path: str
    width: int
    height: int
    mask_area_pct: float


def collect_clean_images(clean_dir: Path) -> List[Path]:
    return sorted(p for p in clean_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def task_id(mask_path: Path) -> int:
    match = TASK_RE.search(mask_path.name)
    if not match:
        raise ValueError(f"Cannot parse task id from mask filename: {mask_path.name}")
    return int(match.group(1))


def collect_masks_by_task(real_masks_dir: Path) -> List[Tuple[int, List[Path]]]:
    groups: Dict[int, List[Path]] = {}
    for p in sorted(real_masks_dir.glob("*.png")):
        groups.setdefault(task_id(p), []).append(p)
    return [(task, sorted(paths)) for task, paths in sorted(groups.items())]


def original_name_from_label_studio_path(value: str) -> str:
    name = Path(value).name
    # Label Studio uploaded filenames often look like
    # "99eab1fb-DSCN5186.JPG"; strip the upload hash prefix.
    if "-" in name:
        prefix, rest = name.split("-", 1)
        if re.fullmatch(r"[0-9a-fA-F]{6,}", prefix):
            return rest
    return name


def normalize_image_name(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = stem.replace("ё", "е")
    stem = stem.replace("х", "x")
    stem = stem.replace("ő", "x")
    return re.sub(r"[^0-9a-zа-я]+", "", stem)


def load_label_studio_mapping(export_json: Path) -> Dict[int, str]:
    data = json.loads(export_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("Expected Label Studio JSON export to be a list of tasks/items")

    mapping: Dict[int, str] = {}
    for item in data:
        task = int(item["id"])

        image_value = None
        if isinstance(item.get("data"), dict):
            image_value = item["data"].get("image") or item["data"].get("img")

        if image_value is None:
            for key, value in item.items():
                if key.isdigit() and isinstance(value, str) and value.lower().endswith(tuple(IMG_EXTS)):
                    image_value = value
                    break

        if image_value is None:
            raise RuntimeError(f"Cannot find source image path for Label Studio task id {task}")

        mapping[task] = original_name_from_label_studio_path(image_value)

    return mapping


def read_binary_mask(path: Path, size: Tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    arr = np.asarray(mask)
    return (arr > 0).astype(np.uint8) * 255


def combine_masks(paths: List[Path], size: Tuple[int, int]) -> np.ndarray:
    combined = np.zeros((size[1], size[0]), dtype=np.uint8)
    for p in paths:
        combined = np.maximum(combined, read_binary_mask(p, size))
    return combined


def make_overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.35) -> Image.Image:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    base = np.asarray(rgb).copy()
    color = np.zeros_like(base)
    color[:, :, 2] = 255
    m = mask > 0
    base[m] = (base[m] * (1 - alpha) + color[m] * alpha).astype(np.uint8)
    return Image.fromarray(base)


def make_debug(image: Image.Image, mask: np.ndarray, overlay: Image.Image) -> Image.Image:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    mask_rgb = Image.fromarray(mask).convert("RGB")

    target_h = 420
    scale = target_h / rgb.height
    target_w = int(rgb.width * scale)
    panels = [
        rgb.resize((target_w, target_h), Image.Resampling.LANCZOS),
        mask_rgb.resize((target_w, target_h), Image.Resampling.NEAREST),
        overlay.resize((target_w, target_h), Image.Resampling.LANCZOS),
    ]
    canvas = Image.new("RGB", (target_w * len(panels), target_h))
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * target_w, 0))
    return canvas


def split_name(i: int, n: int, val_ratio: float) -> str:
    val_count = max(1, round(n * val_ratio)) if n > 1 else 0
    return "val" if i >= n - val_count else "train"


def prepare_dataset(
    clean_dir: Path,
    real_masks_dir: Path,
    out_dir: Path,
    val_ratio: float,
    label_studio_json: Path | None = None,
) -> List[PreparedItem]:
    clean_images = collect_clean_images(clean_dir)
    mask_groups = collect_masks_by_task(real_masks_dir)
    clean_by_name = {p.name: p for p in clean_images}
    clean_by_norm_name = {normalize_image_name(p.name): p for p in clean_images}

    if label_studio_json is not None:
        ls_mapping = load_label_studio_mapping(label_studio_json)
        pairs: List[Tuple[Path, int, List[Path]]] = []
        missing_images = []
        for task, mask_paths in mask_groups:
            image_name = ls_mapping.get(task)
            clean_path = clean_by_name.get(image_name or "")
            if clean_path is None and image_name:
                clean_path = clean_by_norm_name.get(normalize_image_name(image_name))
            if clean_path is None:
                missing_images.append((task, image_name))
                continue
            pairs.append((clean_path, task, mask_paths))

        if missing_images:
            preview = ", ".join(f"{task}:{name}" for task, name in missing_images[:10])
            raise RuntimeError(f"Could not match {len(missing_images)} tasks to clean images: {preview}")
    else:
        if len(clean_images) != len(mask_groups):
            raise RuntimeError(
                f"Clean images count ({len(clean_images)}) does not match unique mask task count ({len(mask_groups)}). "
                f"Raw mask files: {sum(len(paths) for _, paths in mask_groups)}"
            )
        pairs = [(clean_path, task, mask_paths) for clean_path, (task, mask_paths) in zip(clean_images, mask_groups)]

    for sub in [
        "images",
        "masks",
        "debug",
        "overlays",
        "train/images",
        "train/masks",
        "val/images",
        "val/masks",
    ]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    rows: List[PreparedItem] = []
    for i, (clean_path, task, mask_paths) in enumerate(pairs, start=1):
        image = Image.open(clean_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        mask = combine_masks(mask_paths, image.size)
        mask_img = Image.fromarray(mask)
        overlay = make_overlay(image, mask)
        debug = make_debug(image, mask, overlay)

        safe_stem = f"{i:03d}_{clean_path.stem}"
        image_out = out_dir / "images" / f"{safe_stem}.jpg"
        mask_out = out_dir / "masks" / f"{safe_stem}.png"
        overlay_out = out_dir / "overlays" / f"{safe_stem}.jpg"
        debug_out = out_dir / "debug" / f"{safe_stem}.jpg"

        image.save(image_out, quality=95)
        mask_img.save(mask_out)
        overlay.save(overlay_out, quality=95)
        debug.save(debug_out, quality=95)

        split = split_name(i - 1, len(pairs), val_ratio)
        split_image = out_dir / split / "images" / image_out.name
        split_mask = out_dir / split / "masks" / mask_out.name
        shutil.copy2(image_out, split_image)
        shutil.copy2(mask_out, split_mask)

        rows.append(
            PreparedItem(
                index=i,
                split=split,
                clean_path=str(clean_path),
                source_mask_paths=";".join(str(p) for p in mask_paths),
                image_path=str(split_image),
                mask_path=str(split_mask),
                width=image.width,
                height=image.height,
                mask_area_pct=float((mask > 0).mean() * 100.0),
            )
        )

    return rows


def save_manifest(out_dir: Path, rows: List[PreparedItem]) -> None:
    with open(out_dir / "manifest.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PreparedItem.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=Path, required=True)
    parser.add_argument("--real_masks_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("talc_unetpp_dataset"))
    parser.add_argument("--label_studio_json", type=Path, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = prepare_dataset(
        args.clean_dir,
        args.real_masks_dir,
        args.out_dir,
        args.val_ratio,
        label_studio_json=args.label_studio_json,
    )
    save_manifest(args.out_dir, rows)

    train = sum(row.split == "train" for row in rows)
    val = sum(row.split == "val" for row in rows)
    print(f"Prepared pairs: {len(rows)}")
    print(f"Train: {train}, Val: {val}")
    print(f"Dataset: {args.out_dir}")
    print(f"Debug overlays: {args.out_dir / 'debug'}")
    print(f"Manifest: {args.out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
