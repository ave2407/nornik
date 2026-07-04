from __future__ import annotations

import cv2
import numpy as np

from .schemas import MaskStats


def compute_mask_stats(mask: np.ndarray) -> MaskStats:
    binary = mask.astype(np.uint8)
    total_pixels = int(binary.size)
    mask_pixels = int(binary.sum())
    component_count = 0
    largest_pixels = 0
    largest_bbox: list[int] | None = None

    if mask_pixels:
        labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        component_count = int(labels - 1)
        if component_count:
            component_stats = stats[1:]
            largest_idx = int(np.argmax(component_stats[:, cv2.CC_STAT_AREA]))
            x = int(component_stats[largest_idx, cv2.CC_STAT_LEFT])
            y = int(component_stats[largest_idx, cv2.CC_STAT_TOP])
            w = int(component_stats[largest_idx, cv2.CC_STAT_WIDTH])
            h = int(component_stats[largest_idx, cv2.CC_STAT_HEIGHT])
            largest_pixels = int(component_stats[largest_idx, cv2.CC_STAT_AREA])
            largest_bbox = [x, y, x + w, y + h]

    fill_percent = 100.0 * mask_pixels / max(total_pixels, 1)
    return MaskStats(
        total_pixels=total_pixels,
        mask_pixels=mask_pixels,
        fill_percent=fill_percent,
        component_count=component_count,
        largest_component_pixels=largest_pixels,
        largest_component_bbox=largest_bbox,
    )
