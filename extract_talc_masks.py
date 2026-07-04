#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract talc masks from paired clean/annotated ore microscopy images.

Input:
  clean image:     original microscope photo without blue annotation
  annotated image: exact same photo with blue contour lines

Output:
  out_dir/
    masks/        binary talc masks, 0/255 PNG
    blue_lines/   detected blue line masks, 0/255 PNG
    overlays/     clean image + transparent mask for visual QC
    debug/        side-by-side debug images
    qc_report.csv quality-control table

Install:
  pip install opencv-python numpy pandas tqdm

Example:
  python extract_talc_masks.py \
    --clean_dir "Фото руд по сортам. ч1/Оталькованные руды" \
    --annot_dir "Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
    --out_dir "talc_masks_out" \
    --method flood
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass
class QCRow:
    filename: str
    clean_path: str
    annot_path: str
    width: int
    height: int
    blue_line_area_pct: float
    mask_area_pct: float
    components: int
    status: str
    warning: str


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """cv2.imread with Cyrillic/Unicode path support."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    """cv2.imwrite with Cyrillic/Unicode path support."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    ext = ".jpg" if suffix in {".jpeg", ".jpg"} else suffix
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    buf.tofile(str(path))


def normalize_name(path: Path) -> str:
    """
    Normalizes filenames for pairing.
    Keeps stem meaningful, removes spaces and common extra words.
    """
    stem = path.stem.lower()
    stem = stem.replace("ё", "е")
    stem = re.sub(r"\s+", "", stem)
    stem = stem.replace("областиоталькования", "")
    stem = stem.replace("оталькование", "")
    stem = stem.replace("разметка", "")
    stem = stem.replace("annotated", "")
    stem = stem.replace("annotation", "")
    stem = stem.replace("mask", "")
    stem = stem.replace("_", "")
    stem = stem.replace("-", "")
    return stem


def collect_images(root: Path, exclude_dir: Optional[Path] = None) -> List[Path]:
    exclude_resolved = exclude_dir.resolve() if exclude_dir is not None else None
    images: List[Path] = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        if exclude_resolved is not None:
            try:
                p.resolve().relative_to(exclude_resolved)
                continue
            except ValueError:
                pass
        images.append(p)
    return sorted(images)


def make_clean_index(clean_dir: Path, annot_dir: Path) -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = {}
    for p in collect_images(clean_dir, exclude_dir=annot_dir):
        key = normalize_name(p)
        idx.setdefault(key, []).append(p)
    return idx


def pair_images(clean_dir: Path, annot_dir: Path) -> List[Tuple[Path, Path]]:
    clean_idx = make_clean_index(clean_dir, annot_dir)
    pairs: List[Tuple[Path, Path]] = []

    for annot_path in collect_images(annot_dir):
        key = normalize_name(annot_path)
        candidates = clean_idx.get(key, [])

        # exact normalized match
        if len(candidates) == 1:
            pairs.append((candidates[0], annot_path))
            continue

        # fallback: exact filename match
        exact = [p for p in collect_images(clean_dir, exclude_dir=annot_dir) if p.name == annot_path.name]
        if len(exact) == 1:
            pairs.append((exact[0], annot_path))
            continue

        print(f"[WARN] no unique clean pair for: {annot_path}")

    return pairs


def odd_kernel(k: int) -> int:
    k = int(k)
    if k < 1:
        return 1
    return k if k % 2 == 1 else k + 1


def auto_kernel_size(h: int, w: int, ratio: float, min_k: int, max_k: int) -> int:
    k = int(round(min(h, w) * ratio))
    k = max(min_k, min(max_k, k))
    return odd_kernel(k)


def extract_blue_line_mask(
    annot_bgr: np.ndarray,
    clean_bgr: Optional[np.ndarray] = None,
    diff_thresh: int = 25,
) -> np.ndarray:
    """
    Detect blue annotation line.
    Uses HSV blue threshold + optional exact-pair difference with clean image.

    OpenCV HSV hue range: 0..179.
    Blue line usually has high saturation and hue around 105..130.
    """
    hsv = cv2.cvtColor(annot_bgr, cv2.COLOR_BGR2HSV)

    # Main blue threshold.
    # If it misses some blue pixels, widen hue range or lower saturation.
    lower_blue = np.array([90, 70, 40], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_blue, upper_blue)

    # Blue dominance in BGR space: B must be noticeably higher than R/G.
    b, g, r = cv2.split(annot_bgr)
    blue_dom = ((b.astype(np.int16) > r.astype(np.int16) + 25) &
                (b.astype(np.int16) > g.astype(np.int16) + 10) &
                (b > 80)).astype(np.uint8) * 255

    mask = cv2.bitwise_and(mask_hsv, blue_dom)

    # Since clean and annotated are exact copies, difference helps catch antialiasing.
    if clean_bgr is not None and clean_bgr.shape == annot_bgr.shape:
        diff = cv2.absdiff(annot_bgr, clean_bgr)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_mask = (diff_gray > diff_thresh).astype(np.uint8) * 255

        # Keep only changed pixels that are blue-ish in annotated image.
        diff_blue = cv2.bitwise_and(diff_mask, blue_dom)
        mask = cv2.bitwise_or(mask, diff_blue)

    # Remove tiny noise.
    small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_k, iterations=1)

    return mask


def thicken_and_close_line(
    blue_mask: np.ndarray,
    dilate_kernel: Optional[int] = None,
    close_kernel: Optional[int] = None,
    dilate_iter: int = 1,
    close_iter: int = 1,
) -> np.ndarray:
    h, w = blue_mask.shape[:2]

    if dilate_kernel is None:
        # approx 0.25% of min dimension, for 2272x1704 -> about 5 px
        dilate_kernel = auto_kernel_size(h, w, ratio=0.0025, min_k=3, max_k=11)
    else:
        dilate_kernel = odd_kernel(dilate_kernel)

    if close_kernel is None:
        # approx 0.8% of min dimension, for 2272x1704 -> about 13 px
        close_kernel = auto_kernel_size(h, w, ratio=0.008, min_k=7, max_k=31)
    else:
        close_kernel = odd_kernel(close_kernel)

    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))

    wall = cv2.dilate(blue_mask, k_dil, iterations=dilate_iter)
    wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, k_close, iterations=close_iter)
    wall = (wall > 0).astype(np.uint8) * 255
    return wall


def connect_line_to_nearby_borders(
    wall: np.ndarray,
    border_gap: Optional[int] = None,
    bridge_thickness: Optional[int] = None,
) -> np.ndarray:
    """
    Close small gaps between annotation lines and image borders.

    Flood filling treats any gap to the border as an opening. Some hand-drawn
    contours visually reach the edge but stop a few pixels short, so we extend
    only connected line components whose bounding box is already near a border.
    """
    h, w = wall.shape[:2]
    if border_gap is None:
        border_gap = auto_kernel_size(h, w, ratio=0.02, min_k=15, max_k=75)
    if border_gap <= 0:
        return wall

    if bridge_thickness is None:
        bridge_thickness = auto_kernel_size(h, w, ratio=0.0025, min_k=3, max_k=11)
    else:
        bridge_thickness = odd_kernel(bridge_thickness)

    out = wall.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats((wall > 0).astype(np.uint8), 8)

    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        x2 = x + cw - 1
        y2 = y + ch - 1

        ys, xs = np.where(labels == i)
        if xs.size == 0:
            continue

        if x <= border_gap:
            idx = int(np.argmin(xs))
            cv2.line(out, (0, int(ys[idx])), (int(xs[idx]), int(ys[idx])), 255, bridge_thickness)
        if w - 1 - x2 <= border_gap:
            idx = int(np.argmax(xs))
            cv2.line(out, (int(xs[idx]), int(ys[idx])), (w - 1, int(ys[idx])), 255, bridge_thickness)
        if y <= border_gap:
            idx = int(np.argmin(ys))
            cv2.line(out, (int(xs[idx]), 0), (int(xs[idx]), int(ys[idx])), 255, bridge_thickness)
        if h - 1 - y2 <= border_gap:
            idx = int(np.argmax(ys))
            cv2.line(out, (int(xs[idx]), int(ys[idx])), (int(xs[idx]), h - 1), 255, bridge_thickness)

    return (out > 0).astype(np.uint8) * 255


def fill_by_flood(wall: np.ndarray) -> np.ndarray:
    """
    Safe method:
    - blue line is treated as a wall
    - only fully closed regions are filled
    - open contours are NOT aggressively hallucinated
    """
    h, w = wall.shape[:2]

    # Free space is 255, wall is 0.
    free = cv2.bitwise_not(wall)
    flood = free.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Flood-fill all border-connected background.
    for x in range(w):
        if flood[0, x] == 255:
            cv2.floodFill(flood, ff_mask, (x, 0), 0)
        if flood[h - 1, x] == 255:
            cv2.floodFill(flood, ff_mask, (x, h - 1), 0)

    for y in range(h):
        if flood[y, 0] == 255:
            cv2.floodFill(flood, ff_mask, (0, y), 0)
        if flood[y, w - 1] == 255:
            cv2.floodFill(flood, ff_mask, (w - 1, y), 0)

    # Remaining white free-space pixels are holes enclosed by blue walls.
    holes = flood
    mask = holes.copy()

    # Include boundary line into target mask.
    # If you want only inside area, comment this line.
    mask[wall > 0] = 255

    return (mask > 0).astype(np.uint8) * 255


def fill_by_contours(wall: np.ndarray, min_area: int) -> np.ndarray:
    """
    More aggressive method:
    Fills external contours of thick blue lines.
    Can work better for almost-closed shapes, but may overfill open curves.
    """
    contours, _ = cv2.findContours(wall, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(wall)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return (mask > 0).astype(np.uint8) * 255


def remove_small_components(mask: np.ndarray, min_area: int) -> Tuple[np.ndarray, int]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask)
    kept = 0

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == i] = 255
            kept += 1

    return out, kept


def make_overlay(clean_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    overlay = clean_bgr.copy()
    color = np.zeros_like(clean_bgr)
    # Blue fill in BGR.
    color[:, :, 0] = 255
    m = mask > 0
    overlay[m] = cv2.addWeighted(clean_bgr[m], 1 - alpha, color[m], alpha, 0)
    return overlay


def make_debug(clean: np.ndarray, annot: np.ndarray, blue: np.ndarray, mask: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    def to_bgr(x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        return x

    h, w = clean.shape[:2]
    target_h = 380
    scale = target_h / h
    target_w = int(w * scale)

    imgs = [
        clean,
        annot,
        to_bgr(blue),
        to_bgr(mask),
        overlay,
    ]
    resized = [cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA) for img in imgs]
    canvas = np.concatenate(resized, axis=1)

    labels = ["clean", "annotated", "wall", "filled mask", "overlay"]
    x = 10
    for lab in labels:
        cv2.putText(canvas, lab, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, lab, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        x += target_w

    return canvas


def process_pair(
    clean_path: Path,
    annot_path: Path,
    out_dir: Path,
    method: str,
    min_area: int,
    dilate_kernel: Optional[int],
    close_kernel: Optional[int],
    dilate_iter: int,
    close_iter: int,
    border_gap: Optional[int],
    diff_thresh: int,
) -> QCRow:
    clean = imread_unicode(clean_path)
    annot = imread_unicode(annot_path)

    if clean is None or annot is None:
        return QCRow(
            filename=annot_path.name,
            clean_path=str(clean_path),
            annot_path=str(annot_path),
            width=0,
            height=0,
            blue_line_area_pct=0.0,
            mask_area_pct=0.0,
            components=0,
            status="error",
            warning="could_not_read_image",
        )

    if clean.shape != annot.shape:
        # Resize annotated to clean only as a fallback; ideally this should never happen.
        annot = cv2.resize(annot, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_LINEAR)
        shape_warning = "shape_mismatch_resized"
    else:
        shape_warning = ""

    h, w = clean.shape[:2]
    blue = extract_blue_line_mask(annot, clean, diff_thresh=diff_thresh)
    wall = thicken_and_close_line(
        blue,
        dilate_kernel=dilate_kernel,
        close_kernel=close_kernel,
        dilate_iter=dilate_iter,
        close_iter=close_iter,
    )
    wall = connect_line_to_nearby_borders(wall, border_gap=border_gap)

    if method == "flood":
        mask = fill_by_flood(wall)
    elif method == "contour":
        mask = fill_by_contours(wall, min_area=min_area)
    else:
        raise ValueError(f"Unknown method: {method}")

    mask, components = remove_small_components(mask, min_area=min_area)

    overlay = make_overlay(clean, mask)
    debug = make_debug(clean, annot, wall, mask, overlay)

    stem = clean_path.stem
    imwrite_unicode(out_dir / "masks" / f"{stem}.png", mask)
    imwrite_unicode(out_dir / "blue_lines" / f"{stem}.png", blue)
    imwrite_unicode(out_dir / "overlays" / f"{stem}.jpg", overlay)
    imwrite_unicode(out_dir / "debug" / f"{stem}.jpg", debug)

    blue_pct = float((blue > 0).mean() * 100.0)
    mask_pct = float((mask > 0).mean() * 100.0)

    warnings = []
    if shape_warning:
        warnings.append(shape_warning)
    if blue_pct < 0.01:
        warnings.append("almost_no_blue_detected")
    if mask_pct < 0.1:
        warnings.append("almost_empty_mask")
    if mask_pct > 80:
        warnings.append("suspiciously_large_mask")
    if components == 0:
        warnings.append("no_components_after_filter")

    status = "ok" if not warnings else "check"

    return QCRow(
        filename=clean_path.name,
        clean_path=str(clean_path),
        annot_path=str(annot_path),
        width=w,
        height=h,
        blue_line_area_pct=blue_pct,
        mask_area_pct=mask_pct,
        components=components,
        status=status,
        warning=";".join(warnings),
    )


def save_qc_report(out_dir: Path, rows: List[QCRow]) -> None:
    path = out_dir / "qc_report.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(QCRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=Path, required=True, help="Folder with clean original images")
    parser.add_argument("--annot_dir", type=Path, required=True, help="Folder with same images but with blue contour annotation")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output folder")

    parser.add_argument(
        "--method",
        choices=["flood", "contour"],
        default="flood",
        help=(
            "flood = safer, fills only closed regions. "
            "contour = more aggressive, can fill almost-open curves but may overfill."
        ),
    )
    parser.add_argument("--min_area", type=int, default=1000, help="Remove mask components smaller than this many pixels")
    parser.add_argument("--diff_thresh", type=int, default=25, help="Pixel difference threshold for pair-based blue detection")

    parser.add_argument("--dilate_kernel", type=int, default=None, help="Line dilation kernel, odd px. Default: auto")
    parser.add_argument("--close_kernel", type=int, default=None, help="Line closing kernel, odd px. Default: auto")
    parser.add_argument(
        "--border_gap",
        type=int,
        default=None,
        help="Connect line components to image borders when the gap is <= this many px. Default: auto, 0 disables",
    )
    parser.add_argument("--dilate_iter", type=int, default=1)
    parser.add_argument("--close_iter", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["masks", "blue_lines", "overlays", "debug"]:
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)

    pairs = pair_images(args.clean_dir, args.annot_dir)
    print(f"Found pairs: {len(pairs)}")

    if not pairs:
        raise RuntimeError("No clean/annotated pairs found. Check folder paths and filenames.")

    rows: List[QCRow] = []
    for clean_path, annot_path in tqdm(pairs):
        row = process_pair(
            clean_path=clean_path,
            annot_path=annot_path,
            out_dir=args.out_dir,
            method=args.method,
            min_area=args.min_area,
            dilate_kernel=args.dilate_kernel,
            close_kernel=args.close_kernel,
            dilate_iter=args.dilate_iter,
            close_iter=args.close_iter,
            border_gap=args.border_gap,
            diff_thresh=args.diff_thresh,
        )
        rows.append(row)

    save_qc_report(args.out_dir, rows)

    ok = sum(r.status == "ok" for r in rows)
    check = sum(r.status == "check" for r in rows)
    err = sum(r.status == "error" for r in rows)

    print(f"Done.")
    print(f"OK: {ok}, CHECK: {check}, ERROR: {err}")
    print(f"QC report: {args.out_dir / 'qc_report.csv'}")
    print(f"Review debug images in: {args.out_dir / 'debug'}")


if __name__ == "__main__":
    main()
