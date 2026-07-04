from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
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


def require_project(project_id: str) -> ProjectInfo:
    try:
        return get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc


def run_inference_job(project_id: str) -> None:
    try:
        set_status(project_id, "running", error=None)
        root = project_dir(project_id)

        def update_progress(done: int, total: int) -> None:
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
        final_mask = read_mask(root / "mask_final.png")
        make_overlay(source_path(project_id), final_mask, root / "overlay_preview.jpg")
        meta["threshold"] = get_project(project_id).threshold
        write_json(root / "inference_meta.json", meta)
        update_project(project_id, inference_progress={**meta, "percent": 100.0})
        set_status(project_id, "ready")
    except Exception as exc:  # noqa: BLE001
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
    update_project(
        project_id,
        status="running",
        error=None,
        inference_progress={"processed_tiles": 0, "total_tiles": None, "percent": 0.0},
    )
    inference_executor.submit(run_inference_job, project_id)
    return get_project(project_id)


@app.patch("/api/projects/{project_id}/threshold", response_model=ProjectInfo)
def api_update_threshold(project_id: str, payload: ThresholdUpdate) -> ProjectInfo:
    require_project(project_id)
    update_project(project_id, threshold=payload.threshold)
    recompute_final_mask(project_id)
    root = project_dir(project_id)
    final_mask = read_mask(root / "mask_final.png")
    make_overlay(source_path(project_id), final_mask, root / "overlay_preview.jpg")
    return get_project(project_id)


@app.post("/api/projects/{project_id}/edits", response_model=ProjectInfo)
def api_apply_edit(project_id: str, payload: StrokeEdit) -> ProjectInfo:
    require_project(project_id)
    root = project_dir(project_id)
    target = root / ("mask_add.png" if payload.mode == "add" else "mask_erase.png")
    layer = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
    if layer is None:
        raise HTTPException(status_code=404, detail="Edit layer not found")

    points = np.array(payload.points, dtype=np.int32)
    if len(points) == 1:
        cv2.circle(layer, tuple(points[0]), payload.radius, 255, thickness=-1)
    elif len(points) > 1:
        cv2.polylines(layer, [points], isClosed=False, color=255, thickness=payload.radius * 2)
        for point in points:
            cv2.circle(layer, tuple(point), payload.radius, 255, thickness=-1)

    cv2.imwrite(str(target), layer)
    recompute_final_mask(project_id)
    final_mask = read_mask(root / "mask_final.png")
    make_overlay(source_path(project_id), final_mask, root / "overlay_preview.jpg")
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


def make_tile(image: np.ndarray, z: int, x: int, y: int, is_mask: bool) -> bytes:
    height, width = image.shape[:2]
    max_level = int(np.ceil(np.log2(max(width, height) / VIEW_TILE_SIZE)))
    max_level = max(max_level, 0)
    scale = 2 ** max(max_level - z, 0)
    tile_size = VIEW_TILE_SIZE * scale
    x0 = x * tile_size
    y0 = y * tile_size
    crop = image[y0 : y0 + tile_size, x0 : x0 + tile_size]
    if crop.size == 0:
        crop = np.zeros((1, 1), dtype=np.uint8) if is_mask else np.zeros((1, 1, 3), dtype=np.uint8)
    if scale != 1:
        crop = cv2.resize(crop, (VIEW_TILE_SIZE, VIEW_TILE_SIZE), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA)
    ext = ".png" if is_mask else ".jpg"
    ok, encoded = cv2.imencode(ext, crop)
    if not ok:
        raise HTTPException(status_code=500, detail="Tile encoding failed")
    return encoded.tobytes()


@app.get("/api/projects/{project_id}/tiles/image/{z}/{x}/{y}.jpg")
def api_image_tile(project_id: str, z: int, x: int, y: int) -> Response:
    require_project(project_id)
    image = cv2.imread(str(source_path(project_id)), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=404, detail="Source image not found")
    return Response(make_tile(image, z, x, y, is_mask=False), media_type="image/jpeg")


@app.get("/api/projects/{project_id}/tiles/mask/{z}/{x}/{y}.png")
def api_mask_tile(project_id: str, z: int, x: int, y: int) -> Response:
    require_project(project_id)
    mask = cv2.imread(str(project_dir(project_id) / "mask_final.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise HTTPException(status_code=404, detail="Mask not found")
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., 2] = 255
    rgba[..., 3] = (mask > 127).astype(np.uint8) * 150
    return Response(make_tile(rgba, z, x, y, is_mask=True), media_type="image/png")
