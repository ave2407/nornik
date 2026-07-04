from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import DEFAULT_THRESHOLD, PROJECTS_ROOT
from .schemas import ImageInfo, MaskStats, ProjectInfo, ProjectStatus
from .stats import compute_mask_stats


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_roots() -> None:
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)


def project_dir(project_id: str) -> Path:
    return PROJECTS_ROOT / project_id


def metadata_path(project_id: str) -> Path:
    return project_dir(project_id) / "project.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def source_path(project_id: str) -> Path:
    meta = read_json(metadata_path(project_id))
    return project_dir(project_id) / meta["source_name"]


def create_project(filename: str, content: bytes) -> ProjectInfo:
    ensure_roots()
    project_id = uuid.uuid4().hex[:12]
    root = project_dir(project_id)
    root.mkdir(parents=True, exist_ok=False)
    (root / "exports").mkdir()

    suffix = Path(filename).suffix.lower() or ".png"
    src_name = f"source{suffix}"
    src = root / src_name
    src.write_bytes(content)

    with Image.open(src) as image:
        width, height = image.size

    empty = np.zeros((height, width), dtype=np.uint8)
    cv2.imwrite(str(root / "mask_add.png"), empty)
    cv2.imwrite(str(root / "mask_erase.png"), empty)
    cv2.imwrite(str(root / "mask_base.png"), empty)
    cv2.imwrite(str(root / "mask_final.png"), empty)

    now = utc_now()
    meta = {
        "id": project_id,
        "status": "created",
        "threshold": DEFAULT_THRESHOLD,
        "source_name": src_name,
        "original_filename": filename,
        "width": width,
        "height": height,
        "created_at": now,
        "updated_at": now,
        "error": None,
        "inference_progress": None,
    }
    write_json(root / "project.json", meta)
    write_stats(project_id, compute_mask_stats(empty))
    return get_project(project_id)


def list_projects() -> list[ProjectInfo]:
    ensure_roots()
    projects = []
    for path in sorted(PROJECTS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.is_dir() and (path / "project.json").exists():
            projects.append(get_project(path.name))
    return projects


def get_project(project_id: str) -> ProjectInfo:
    meta = read_json(metadata_path(project_id))
    stats = None
    stats_path = project_dir(project_id) / "stats.json"
    if stats_path.exists():
        stats = MaskStats(**read_json(stats_path))
    return ProjectInfo(
        id=meta["id"],
        status=meta["status"],
        threshold=float(meta["threshold"]),
        image=ImageInfo(
            width=int(meta["width"]),
            height=int(meta["height"]),
            filename=meta.get("original_filename", meta["source_name"]),
        ),
        stats=stats,
        error=meta.get("error"),
        inference_progress=meta.get("inference_progress"),
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
    )


def update_project(project_id: str, **updates: object) -> None:
    path = metadata_path(project_id)
    meta = read_json(path)
    meta.update(updates)
    meta["updated_at"] = utc_now()
    write_json(path, meta)


def set_status(project_id: str, status: ProjectStatus, error: str | None = None) -> None:
    update_project(project_id, status=status, error=error)


def write_stats(project_id: str, stats: MaskStats) -> None:
    if hasattr(stats, "model_dump"):
        data = stats.model_dump()
    else:
        data = stats.dict()
    write_json(project_dir(project_id) / "stats.json", data)


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 127


def write_mask(path: Path, mask: np.ndarray) -> None:
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def recompute_final_mask(project_id: str) -> MaskStats:
    root = project_dir(project_id)
    meta = read_json(root / "project.json")
    threshold = float(meta["threshold"])
    prob_path = root / "probability.npy"
    if prob_path.exists():
        probability = np.load(prob_path)
        base = probability >= threshold
    else:
        base = read_mask(root / "mask_base.png")

    add = read_mask(root / "mask_add.png")
    erase = read_mask(root / "mask_erase.png")
    final = np.logical_and(np.logical_or(base, add), np.logical_not(erase))
    write_mask(root / "mask_base.png", base)
    write_mask(root / "mask_final.png", final)
    stats = compute_mask_stats(final)
    write_stats(project_id, stats)
    return stats


def save_probability(project_id: str, probability: np.ndarray) -> None:
    np.save(project_dir(project_id) / "probability.npy", probability.astype(np.float32))


def save_source_copy(project_id: str, dst: Path) -> None:
    shutil.copy2(source_path(project_id), dst)
