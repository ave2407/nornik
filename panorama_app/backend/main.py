from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from .classifier import DummyClassifierService
from .config import (
    ALLOWED_IMAGE_EXTS,
    DEFAULT_THRESHOLD,
    INFER_BATCH_SIZE,
    OVERLAP,
    TILE_SIZE,
    VIEW_TILE_SIZE,
)
from .exporter import export_single
from .inference import OnnxSegmentationModel, infer_image_probability, make_overlay
from .schemas import (
    ClassificationResult,
    ExportRequest,
    ProjectInfo,
    StrokeEdit,
    ThresholdUpdate,
)
from .storage import (
    create_project,
    get_project,
    list_projects,
    project_dir,
    read_mask,
    recompute_final_mask,
    save_probability,
    set_status,
    source_path,
    update_project,
    write_json,
    write_mask,
)


app = FastAPI(title="Panorama Talc Mask Editor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = DummyClassifierService()
onnx_model = OnnxSegmentationModel()
inference_executor = ThreadPoolExecutor(max_workers=1)
cancelled_projects: set[str] = set()
OVERLAY_MAX_PIXELS = 20_000_000


@lru_cache(maxsize=6)
def cached_imread(path: str, mtime_ns: int, flags: int) -> np.ndarray | None:
    return cv2.imread(path, flags)


def read_cached_image(path: Path, flags: int) -> np.ndarray | None:
    stat = path.stat()
    image = cached_imread(str(path), stat.st_mtime_ns, flags)
    return None if image is None else image.copy()


def clear_image_cache() -> None:
    cached_imread.cache_clear()


def require_project(project_id: str) -> ProjectInfo:
    try:
        return get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc


def maybe_make_overlay(project_id: str) -> None:
    project = get_project(project_id)
    if project.image.width * project.image.height > OVERLAY_MAX_PIXELS:
        return
    root = project_dir(project_id)
    final_mask = read_mask(root / "mask_final.png")
    make_overlay(source_path(project_id), final_mask, root / "overlay_preview.jpg")


def run_inference_job(project_id: str) -> None:
    try:
        if project_id in cancelled_projects:
            set_status(project_id, "cancelled", error="Cancelled before start")
            return
        set_status(project_id, "running", error=None)
        root = project_dir(project_id)

        def update_progress(done: int, total: int) -> None:
            if project_id in cancelled_projects:
                raise RuntimeError("Inference cancelled")
            update_project(
                project_id,
                inference_progress={
                    "processed_tiles": done,
                    "total_tiles": total,
                    "percent": 100.0 * done / max(total, 1),
                },
            )

        probability, meta = infer_image_probability(
            source_path(project_id),
            onnx_model,
            tile_size=TILE_SIZE,
            overlap=OVERLAP,
            batch_size=INFER_BATCH_SIZE,
            progress_callback=update_progress,
        )
        save_probability(project_id, probability)
        stats = recompute_final_mask(project_id)
        clear_image_cache()
        maybe_make_overlay(project_id)
        meta["threshold"] = get_project(project_id).threshold
        write_json(root / "inference_meta.json", meta)
        update_project(project_id, inference_progress={**meta, "percent": 100.0})
        set_status(project_id, "ready")
    except Exception as exc:  # noqa: BLE001
        if project_id in cancelled_projects:
            set_status(project_id, "cancelled", error="Inference cancelled")
            cancelled_projects.discard(project_id)
        else:
            set_status(project_id, "failed", error=str(exc))


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "onnx_model_ready": onnx_model.ready, "model_path": str(onnx_model.model_path)}


@app.get("/api/projects", response_model=list[ProjectInfo])
def api_list_projects() -> list[ProjectInfo]:
    return list_projects()


@app.post("/api/projects", response_model=ProjectInfo)
async def api_create_project(file: UploadFile = File(...)) -> ProjectInfo:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported image extension: {suffix}")
    content = await file.read()
    return create_project(file.filename or f"upload{suffix}", content)


@app.get("/api/projects/{project_id}", response_model=ProjectInfo)
def api_get_project(project_id: str) -> ProjectInfo:
    return require_project(project_id)


@app.post("/api/projects/{project_id}/infer", response_model=ProjectInfo)
def api_infer_project(project_id: str) -> ProjectInfo:
    require_project(project_id)
    cancelled_projects.discard(project_id)
    update_project(
        project_id,
        status="running",
        error=None,
        inference_progress={"processed_tiles": 0, "total_tiles": None, "percent": 0.0},
    )
    inference_executor.submit(run_inference_job, project_id)
    return get_project(project_id)


@app.post("/api/projects/{project_id}/cancel", response_model=ProjectInfo)
def api_cancel_project(project_id: str) -> ProjectInfo:
    require_project(project_id)
    cancelled_projects.add(project_id)
    set_status(project_id, "cancelled", error="Cancelled by user")
    return get_project(project_id)


@app.post("/api/projects/{project_id}/reset", response_model=ProjectInfo)
def api_reset_project(project_id: str) -> ProjectInfo:
    project = require_project(project_id)
    if project.status != "ready":
        raise HTTPException(status_code=409, detail="Mask is not ready yet")
    root = project_dir(project_id)
    empty = np.zeros((project.image.height, project.image.width), dtype=np.uint8)
    cv2.imwrite(str(root / "mask_add.png"), empty)
    cv2.imwrite(str(root / "mask_erase.png"), empty)
    update_project(project_id, threshold=DEFAULT_THRESHOLD, error=None)
    recompute_final_mask(project_id)
    clear_image_cache()
    maybe_make_overlay(project_id)
    return get_project(project_id)


@app.patch("/api/projects/{project_id}/threshold", response_model=ProjectInfo)
def api_update_threshold(project_id: str, payload: ThresholdUpdate) -> ProjectInfo:
    project = require_project(project_id)
    if project.status != "ready":
        raise HTTPException(status_code=409, detail="Mask is not ready yet")
    update_project(project_id, threshold=payload.threshold)
    recompute_final_mask(project_id)
    clear_image_cache()
    maybe_make_overlay(project_id)
    return get_project(project_id)


@app.post("/api/projects/{project_id}/edits", response_model=ProjectInfo)
def api_apply_edit(project_id: str, payload: StrokeEdit) -> ProjectInfo:
    project = require_project(project_id)
    if project.status != "ready":
        raise HTTPException(status_code=409, detail="Mask is not ready yet")
    root = project_dir(project_id)
    add_path = root / "mask_add.png"
    erase_path = root / "mask_erase.png"
    add_layer = cv2.imread(str(add_path), cv2.IMREAD_GRAYSCALE)
    erase_layer = cv2.imread(str(erase_path), cv2.IMREAD_GRAYSCALE)
    if add_layer is None or erase_layer is None:
        raise HTTPException(status_code=404, detail="Edit layer not found")

    points = np.array(payload.points, dtype=np.int32)
    stroke = np.zeros_like(add_layer)
    if len(points) == 1:
        cv2.circle(stroke, tuple(points[0]), payload.radius, 255, thickness=-1)
    elif len(points) > 1:
        cv2.polylines(stroke, [points], isClosed=False, color=255, thickness=payload.radius * 2)
        for point in points:
            cv2.circle(stroke, tuple(point), payload.radius, 255, thickness=-1)

    stroke_bool = stroke > 0
    if payload.mode == "add":
        add_layer[stroke_bool] = 255
        erase_layer[stroke_bool] = 0
    else:
        erase_layer[stroke_bool] = 255
        add_layer[stroke_bool] = 0

    cv2.imwrite(str(add_path), add_layer)
    cv2.imwrite(str(erase_path), erase_layer)
    recompute_final_mask(project_id)
    clear_image_cache()
    maybe_make_overlay(project_id)
    return get_project(project_id)


@app.post("/api/projects/{project_id}/export")
def api_export_project(project_id: str, payload: ExportRequest) -> FileResponse:
    require_project(project_id)
    path = export_single(project_id, payload.kind)
    return FileResponse(path, filename=path.name)


@app.get("/api/classification/{project_id}", response_model=ClassificationResult)
def api_classification(project_id: str) -> ClassificationResult:
    require_project(project_id)
    return classifier.classify(project_id)


@app.get("/api/projects/{project_id}/source")
def api_source(project_id: str) -> FileResponse:
    require_project(project_id)
    return FileResponse(source_path(project_id))


@app.get("/api/projects/{project_id}/mask")
def api_mask(project_id: str) -> FileResponse:
    require_project(project_id)
    return FileResponse(project_dir(project_id) / "mask_final.png")


@app.get("/api/projects/{project_id}/overlay")
def api_overlay(project_id: str) -> FileResponse:
    require_project(project_id)
    path = project_dir(project_id) / "overlay_preview.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Overlay is not ready")
    return FileResponse(path)


def tile_geometry(image: np.ndarray, z: int, x: int, y: int) -> tuple[slice, slice, int]:
    height, width = image.shape[:2]
    max_level = int(np.ceil(np.log2(max(width, height) / VIEW_TILE_SIZE)))
    max_level = max(max_level, 0)
    scale = 2 ** max(max_level - z, 0)
    tile_size = VIEW_TILE_SIZE * scale
    x0 = x * tile_size
    y0 = y * tile_size
    return slice(y0, min(y0 + tile_size, height)), slice(x0, min(x0 + tile_size, width)), scale


def encode_tile(crop: np.ndarray, scale: int, is_mask: bool) -> bytes:
    if crop.size == 0:
        crop = np.zeros((1, 1), dtype=np.uint8) if is_mask else np.zeros((1, 1, 3), dtype=np.uint8)
    target_w = max(1, int(np.ceil(crop.shape[1] / scale)))
    target_h = max(1, int(np.ceil(crop.shape[0] / scale)))
    crop = cv2.resize(
        crop,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA,
    )
    if crop.ndim == 3:
        canvas = np.zeros((VIEW_TILE_SIZE, VIEW_TILE_SIZE, crop.shape[2]), dtype=np.uint8)
    else:
        canvas = np.zeros((VIEW_TILE_SIZE, VIEW_TILE_SIZE), dtype=np.uint8)
    canvas[: crop.shape[0], : crop.shape[1]] = crop
    ext = ".png" if is_mask else ".jpg"
    ok, encoded = cv2.imencode(ext, canvas)
    if not ok:
        raise HTTPException(status_code=500, detail="Tile encoding failed")
    return encoded.tobytes()


@app.get("/api/projects/{project_id}/tiles/image/{z}/{x}/{y}.jpg")
def api_image_tile(project_id: str, z: int, x: int, y: int) -> Response:
    require_project(project_id)
    image = read_cached_image(source_path(project_id), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=404, detail="Source image not found")
    y_slice, x_slice, scale = tile_geometry(image, z, x, y)
    crop = image[y_slice, x_slice]
    return Response(encode_tile(crop, scale, is_mask=False), media_type="image/jpeg")


@app.get("/api/projects/{project_id}/tiles/mask/{z}/{x}/{y}.png")
def api_mask_tile(project_id: str, z: int, x: int, y: int) -> Response:
    require_project(project_id)
    mask = read_cached_image(project_dir(project_id) / "mask_final.png", cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise HTTPException(status_code=404, detail="Mask not found")
    y_slice, x_slice, scale = tile_geometry(mask, z, x, y)
    mask_crop = mask[y_slice, x_slice]
    rgba = np.zeros((*mask_crop.shape, 4), dtype=np.uint8)
    rgba[..., 2] = 255
    rgba[..., 3] = (mask_crop > 127).astype(np.uint8) * 150
    return Response(encode_tile(rgba, scale, is_mask=True), media_type="image/png")
