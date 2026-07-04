from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MAX_ANALYSIS_SIDE = 4096
TALC_DECISION_THRESHOLD = 10.0


@dataclass
class PhaseAnalysisResult:
    stats: dict
    classification: dict


def _resize_for_analysis(image: np.ndarray, talc_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    height, width = image.shape[:2]
    max_side = max(height, width)
    if max_side <= MAX_ANALYSIS_SIDE:
        return image, talc_mask, 1.0
    scale = MAX_ANALYSIS_SIDE / max_side
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(talc_mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST) > 0
    return resized_image, resized_mask, scale


def _read_talc_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(shape, dtype=bool)
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def _component_masks(sulfide: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    total = int(sulfide.size)
    min_area = max(16, int(total * 0.000005))
    coarse_area = max(1500, int(total * 0.0004))
    labels, label_image, stats, _ = cv2.connectedComponentsWithStats(sulfide.astype(np.uint8), connectivity=8)
    ordinary = np.zeros_like(sulfide, dtype=bool)
    thin = np.zeros_like(sulfide, dtype=bool)
    fine_count = 0
    coarse_count = 0
    largest = 0
    kept = 0
    for idx in range(1, labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        kept += 1
        largest = max(largest, area)
        component = label_image == idx
        if area >= coarse_area:
            ordinary |= component
            coarse_count += 1
        else:
            thin |= component
            fine_count += 1
    return ordinary, thin, {
        "min_component_pixels": min_area,
        "coarse_component_pixels": coarse_area,
        "sulfide_component_count": kept,
        "fine_component_count": fine_count,
        "coarse_component_count": coarse_count,
        "largest_sulfide_component_pixels": largest,
    }


def _detect_sulfides(image_bgr: np.ndarray, talc_mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    luma = lab[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    bright_cut = max(125.0, float(np.percentile(luma, 70)))
    reflective = ((luma >= bright_cut) & (value >= 115)) | ((value >= 155) & (saturation <= 135))
    reflective &= ~talc_mask
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(reflective.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned > 0


def _save_phase_overlay(image_bgr: np.ndarray, talc: np.ndarray, ordinary: np.ndarray, thin: np.ndarray, out: Path) -> None:
    if max(image_bgr.shape[:2]) > MAX_ANALYSIS_SIDE:
        scale = MAX_ANALYSIS_SIDE / max(image_bgr.shape[:2])
        size = (int(round(image_bgr.shape[1] * scale)), int(round(image_bgr.shape[0] * scale)))
        image_bgr = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)
        talc = cv2.resize(talc.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        ordinary = cv2.resize(ordinary.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        thin = cv2.resize(thin.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    overlay = image_bgr.copy()
    color = np.zeros_like(overlay)
    color[talc] = (255, 0, 0)
    color[ordinary] = (0, 180, 0)
    color[thin] = (0, 0, 255)
    active = talc | ordinary | thin
    overlay[active] = cv2.addWeighted(overlay, 0.45, color, 0.55, 0)[active]
    cv2.imwrite(str(out), overlay)


def _classification_from_features(stats: dict) -> dict:
    talc_percent = float(stats["talc_percent"])
    ordinary_area = float(stats["ordinary_intergrowth_area_percent"])
    thin_area = float(stats["thin_intergrowth_area_percent"])
    sulfide = max(float(stats["sulfide_percent"]), 1e-6)

    if talc_percent > TALC_DECISION_THRESHOLD:
        confidence = min(0.98, 0.72 + (talc_percent - TALC_DECISION_THRESHOLD) / 60.0)
        return {
            "class_name": "talc",
            "display_name": "Оталькованная руда",
            "confidence": confidence,
            "probs": {
                "talc": confidence,
                "ordinary": (1.0 - confidence) * 0.45,
                "difficult": (1.0 - confidence) * 0.55,
            },
            "decision_reason": f"talc_percent={talc_percent:.2f}% > 10%, talc rule has priority",
        }

    ordinary_score = ordinary_area + 0.15 * float(stats["coarse_component_count"])
    difficult_score = thin_area * 1.25 + 0.03 * float(stats["fine_component_count"])
    margin = abs(ordinary_score - difficult_score) / max(sulfide, 1.0)
    confidence = min(0.86, 0.55 + margin)
    if ordinary_score >= difficult_score:
        return {
            "class_name": "ordinary",
            "display_name": "Рядовая руда",
            "confidence": confidence,
            "probs": {
                "ordinary": confidence,
                "difficult": max(0.0, 1.0 - confidence - 0.03),
                "talc": 0.03,
            },
            "decision_reason": "coarse ordinary sulfide intergrowth score dominates fine intergrowth score",
        }
    return {
        "class_name": "difficult",
        "display_name": "Труднообогатимая руда",
        "confidence": confidence,
        "probs": {
            "difficult": confidence,
            "ordinary": max(0.0, 1.0 - confidence - 0.03),
            "talc": 0.03,
        },
        "decision_reason": "fine/disseminated sulfide intergrowth score dominates coarse intergrowth score",
    }


def analyze_image(source_path: Path, talc_mask_path: Path, out_dir: Path) -> PhaseAnalysisResult:
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(source_path)
    source_h, source_w = image.shape[:2]
    talc_full = _read_talc_mask(talc_mask_path, (source_h, source_w))
    analysis_image, analysis_talc, scale = _resize_for_analysis(image, talc_full)
    sulfide = _detect_sulfides(analysis_image, analysis_talc)
    ordinary, thin, component_stats = _component_masks(sulfide)
    total = int(analysis_talc.size)
    talc_pixels = int(analysis_talc.sum())
    ordinary_pixels = int(ordinary.sum())
    thin_pixels = int(thin.sum())
    sulfide_pixels = ordinary_pixels + thin_pixels
    gangue_pixels = max(0, total - talc_pixels - sulfide_pixels)
    stats = {
        "source_width": source_w,
        "source_height": source_h,
        "analysis_width": int(analysis_image.shape[1]),
        "analysis_height": int(analysis_image.shape[0]),
        "analysis_scale": float(scale),
        "total_pixels": total,
        "talc_pixels": talc_pixels,
        "sulfide_pixels": sulfide_pixels,
        "gangue_pixels": gangue_pixels,
        "ordinary_intergrowth_pixels": ordinary_pixels,
        "thin_intergrowth_pixels": thin_pixels,
        "talc_percent": 100.0 * talc_pixels / max(total, 1),
        "sulfide_percent": 100.0 * sulfide_pixels / max(total, 1),
        "gangue_percent": 100.0 * gangue_pixels / max(total, 1),
        "ordinary_intergrowth_area_percent": 100.0 * ordinary_pixels / max(total, 1),
        "thin_intergrowth_area_percent": 100.0 * thin_pixels / max(total, 1),
        **component_stats,
    }
    classification = _classification_from_features(stats)
    classification.update(
        {
            "model_version": "expert-rules-v1",
            "rule_version": "talc_gt_10_then_intergrowth_components_v1",
            "phase_stats": stats,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "classification.json").write_text(json.dumps(classification, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "phase_masks.npz",
        talc=analysis_talc.astype(np.uint8),
        ordinary=ordinary.astype(np.uint8),
        thin=thin.astype(np.uint8),
        source_shape=np.array([source_h, source_w], dtype=np.int32),
    )
    _save_phase_overlay(analysis_image, analysis_talc, ordinary, thin, out_dir / "phase_overlay.jpg")
    return PhaseAnalysisResult(stats=stats, classification=classification)
