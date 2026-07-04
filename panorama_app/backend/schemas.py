from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ProjectStatus = Literal["created", "running", "ready", "failed", "cancelled"]
EditMode = Literal["add", "erase"]
ExportKind = Literal[
    "zip",
    "mask_png",
    "overlay_jpg",
    "stats_json",
    "classification_json",
    "phase_stats_json",
    "phase_overlay_jpg",
]


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


class PhaseStats(BaseModel):
    source_width: int
    source_height: int
    analysis_width: int
    analysis_height: int
    analysis_scale: float
    total_pixels: int
    talc_pixels: int
    sulfide_pixels: int
    gangue_pixels: int
    ordinary_intergrowth_pixels: int
    thin_intergrowth_pixels: int
    talc_percent: float
    sulfide_percent: float
    gangue_percent: float
    ordinary_intergrowth_area_percent: float
    thin_intergrowth_area_percent: float
    min_component_pixels: int
    coarse_component_pixels: int
    sulfide_component_count: int
    fine_component_count: int
    coarse_component_count: int
    largest_sulfide_component_pixels: int


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
    display_name: str = "Неизвестно"
    confidence: float | None = None
    probs: dict[str, float] = Field(default_factory=dict)
    model_class_name: str = "unknown"
    model_display_name: str = "Неизвестно"
    model_confidence: float | None = None
    model_probs: dict[str, float] = Field(default_factory=dict)
    model_version: str = "stub"
    rule_version: str = "stub"
    decision_reason: str = "classification model is not connected"
    phase_stats: PhaseStats | None = None
