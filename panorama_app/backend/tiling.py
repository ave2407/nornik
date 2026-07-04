from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileSpec:
    x: int
    y: int
    width: int
    height: int
    padded_x: int
    padded_y: int


def compute_padding(height: int, width: int, tile_size: int, stride: int) -> tuple[int, int]:
    def padded(length: int) -> int:
        if length <= tile_size:
            return tile_size
        steps = int(np.ceil((length - tile_size) / stride))
        return tile_size + steps * stride

    return padded(height), padded(width)


def pad_image_reflect(image: np.ndarray, tile_size: int, stride: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    padded_h, padded_w = compute_padding(height, width, tile_size, stride)
    pad_h = padded_h - height
    pad_w = padded_w - width
    mode = "reflect" if height > 1 and width > 1 else "edge"
    if image.ndim == 2:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode=mode)
    else:
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    return padded, (height, width)


def iter_tiles(image: np.ndarray, tile_size: int, stride: int) -> list[tuple[TileSpec, np.ndarray]]:
    padded, (orig_h, orig_w) = pad_image_reflect(image, tile_size, stride)
    padded_h, padded_w = padded.shape[:2]
    tiles: list[tuple[TileSpec, np.ndarray]] = []
    for y in range(0, padded_h - tile_size + 1, stride):
        for x in range(0, padded_w - tile_size + 1, stride):
            spec = TileSpec(
                x=min(x, orig_w),
                y=min(y, orig_h),
                width=max(0, min(tile_size, orig_w - x)),
                height=max(0, min(tile_size, orig_h - y)),
                padded_x=x,
                padded_y=y,
            )
            tiles.append((spec, padded[y : y + tile_size, x : x + tile_size]))
    return tiles

