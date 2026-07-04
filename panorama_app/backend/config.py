from __future__ import annotations

import os
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("PANORAMA_DATA_ROOT", APP_ROOT / "data"))
PROJECTS_ROOT = DATA_ROOT / "projects"
MODELS_ROOT = Path(os.getenv("PANORAMA_MODELS_ROOT", APP_ROOT / "models"))
DEFAULT_MODEL_PATH = Path(
    os.getenv("TALC_ONNX_MODEL", MODELS_ROOT / "talc_unetpp_effb3_768.onnx")
)

TILE_SIZE = int(os.getenv("PANORAMA_TILE_SIZE", "768"))
OVERLAP = int(os.getenv("PANORAMA_OVERLAP", "192"))
INFER_BATCH_SIZE = int(os.getenv("PANORAMA_BATCH_SIZE", "4"))
DEFAULT_THRESHOLD = float(os.getenv("PANORAMA_THRESHOLD", "0.5"))
VIEW_TILE_SIZE = int(os.getenv("PANORAMA_VIEW_TILE_SIZE", "256"))
MIN_COMPONENT_PIXELS = int(os.getenv("PANORAMA_MIN_COMPONENT_PIXELS", "500"))

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
