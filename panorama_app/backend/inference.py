from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import DEFAULT_MODEL_PATH, INFER_BATCH_SIZE, OVERLAP, TILE_SIZE
from .stitching import hann_window
from .tiling import iter_tiles


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class OnnxSegmentationModel:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            self.session = None
            self.input_name = None
            return

        # Importing torch preloads CUDA/cuDNN shared libraries from the venv wheels.
        # This lets onnxruntime-gpu use CUDAExecutionProvider on servers without
        # system-wide CUDA/cuDNN installs.
        try:
            import torch  # noqa: F401
        except Exception:
            pass

        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    @property
    def ready(self) -> bool:
        return self.session is not None and self.input_name is not None

    def predict_batch(self, batch: np.ndarray) -> np.ndarray:
        if not self.ready:
            raise FileNotFoundError(
                f"ONNX model is not available: {self.model_path}. Export it first."
            )
        logits = self.session.run(None, {self.input_name: batch})[0]
        return sigmoid(logits[:, 0])


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_tile(tile: np.ndarray) -> np.ndarray:
    arr = tile.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)


def infer_image_probability(
    image_path: Path,
    model: OnnxSegmentationModel,
    tile_size: int = TILE_SIZE,
    overlap: int = OVERLAP,
    batch_size: int = INFER_BATCH_SIZE,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, dict]:
    started = time.time()
    image = read_rgb(image_path)
    height, width = image.shape[:2]
    stride = tile_size - overlap
    tiles = iter_tiles(image, tile_size=tile_size, stride=stride)
    accum = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    window = hann_window(tile_size)
    if progress_callback:
        progress_callback(0, len(tiles))

    for start in range(0, len(tiles), batch_size):
        chunk = tiles[start : start + batch_size]
        batch = np.stack([preprocess_tile(tile) for _, tile in chunk]).astype(np.float32)
        probs = model.predict_batch(batch)
        for (spec, _), prob in zip(chunk, probs):
            if spec.width <= 0 or spec.height <= 0:
                continue
            crop_prob = prob[: spec.height, : spec.width].astype(np.float32)
            crop_weight = window[: spec.height, : spec.width]
            y2 = spec.y + spec.height
            x2 = spec.x + spec.width
            accum[spec.y:y2, spec.x:x2] += crop_prob * crop_weight
            weights[spec.y:y2, spec.x:x2] += crop_weight
        if progress_callback:
            progress_callback(min(start + len(chunk), len(tiles)), len(tiles))

    probability = accum / np.maximum(weights, 1e-6)
    meta = {
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "overlap": overlap,
        "stride": stride,
        "batch_size": batch_size,
        "tile_count": len(tiles),
        "model_path": str(model.model_path),
        "elapsed_sec": time.time() - started,
    }
    return probability, meta


def make_overlay(image_path: Path, mask: np.ndarray, output_path: Path, max_side: int = 4096) -> None:
    image = read_rgb(image_path)
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay = image.copy()
    blue = np.zeros_like(overlay)
    blue[..., 2] = 255
    alpha = 0.35
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (
        overlay[mask_bool].astype(np.float32) * (1 - alpha)
        + blue[mask_bool].astype(np.float32) * alpha
    ).astype(np.uint8)
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
