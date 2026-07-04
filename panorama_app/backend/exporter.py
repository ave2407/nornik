from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from .storage import project_dir, source_path


def export_project_zip(project_id: str) -> Path:
    root = project_dir(project_id)
    exports = root / "exports"
    exports.mkdir(exist_ok=True)
    out = exports / "project_export.zip"
    include = [
        "project.json",
        "stats.json",
        "phase_stats.json",
        "classification.json",
        "inference_meta.json",
        "mask_base.png",
        "mask_add.png",
        "mask_erase.png",
        "mask_final.png",
        "phase_overlay.jpg",
        "overlay_preview.jpg",
        "phase_masks.npz",
        "probability.npy",
    ]
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        src = source_path(project_id)
        zf.write(src, arcname=src.name)
        for name in include:
            path = root / name
            if path.exists():
                zf.write(path, arcname=name)
    return out


def export_single(project_id: str, kind: str) -> Path:
    root = project_dir(project_id)
    exports = root / "exports"
    exports.mkdir(exist_ok=True)
    if kind == "zip":
        return export_project_zip(project_id)
    mapping = {
        "mask_png": root / "mask_final.png",
        "overlay_jpg": root / "overlay_preview.jpg",
        "stats_json": root / "stats.json",
        "classification_json": root / "classification.json",
        "phase_stats_json": root / "phase_stats.json",
        "phase_overlay_jpg": root / "phase_overlay.jpg",
    }
    src = mapping[kind]
    dst = exports / src.name
    shutil.copy2(src, dst)
    return dst
