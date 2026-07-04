from __future__ import annotations

import numpy as np

from .tiling import TileSpec


def hann_window(tile_size: int) -> np.ndarray:
    one_dim = np.hanning(tile_size)
    window = np.outer(one_dim, one_dim).astype(np.float32)
    return np.maximum(window, 1e-3)


def stitch_probability(
    tile_probs: list[tuple[TileSpec, np.ndarray]],
    original_shape: tuple[int, int],
    tile_size: int,
) -> np.ndarray:
    height, width = original_shape
    accum = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    window = hann_window(tile_size)

    for spec, prob in tile_probs:
        if spec.width <= 0 or spec.height <= 0:
            continue
        crop_prob = prob[: spec.height, : spec.width].astype(np.float32)
        crop_weight = window[: spec.height, : spec.width]
        y2 = spec.y + spec.height
        x2 = spec.x + spec.width
        accum[spec.y:y2, spec.x:x2] += crop_prob * crop_weight
        weights[spec.y:y2, spec.x:x2] += crop_weight

    return accum / np.maximum(weights, 1e-6)

