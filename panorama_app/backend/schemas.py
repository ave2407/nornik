from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ProjectStatus = Literal["created", "running", "ready", "failed"]
EditMode = Literal["add", "erase"]
ExportKind = Literal["zip", "mask_png", "overlay_jpg", "stats_json"]


class ImageInfo(BaseModel):
    width: int
    height: int
    filename: str


class MaskStats(BaseModel):
    total_pixels: int
    mask_pixels: int
    fill_percent: float
    component_count: int
    largest_component_pixels: int
    largest_component_bbox: list[int] | None = None


class ProjectInfo(BaseModel):
    id: str
    status: ProjectStatus
    threshold: float
    image: ImageInfo
    stats: MaskStats | None = None
    error: str | None = None
    inference_progress: dict | None = None
    created_at: str
    updated_at: str


class ThresholdUpdate(BaseModel):
    threshold: float = Field(ge=0.0, le=1.0)


class StrokeEdit(BaseModel):
    mode: EditMode
    points: list[list[float]]
    radius: int = Field(default=24, ge=1, le=512)


class ExportRequest(BaseModel):
    kind: ExportKind = "zip"


class ClassificationResult(BaseModel):
    project_id: str
    class_name: str = "unknown"
    confidence: float | None = None
    probs: dict[str, float] = Field(default_factory=dict)
    model_version: str = "stub"
