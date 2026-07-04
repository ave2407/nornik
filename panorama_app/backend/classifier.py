from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .config import MODELS_ROOT
from .phase_analysis import TALC_DECISION_THRESHOLD, analyze_image
from .schemas import ClassificationResult
from .storage import project_dir, read_json, source_path, write_json


DISPLAY_NAMES = {
    "ordinary": "Рядовая руда",
    "difficult": "Труднообогатимая руда",
    "talc": "Оталькованная руда",
    "unknown": "Неизвестно",
}


class ClassifierService:
    def classify(self, project_id: str) -> ClassificationResult:
        raise NotImplementedError


class OnnxOreClassifier:
    def __init__(self) -> None:
        self.model_path = Path(MODELS_ROOT) / "ore_classifier_3class_effb3.onnx"
        self.meta_path = Path(MODELS_ROOT) / "ore_classifier_3class_effb3.json"
        self.ready = self.model_path.exists() and self.meta_path.exists()
        self.session: ort.InferenceSession | None = None
        self.input_name = "image"
        self.output_name = "logits"
        self.classes = ["ordinary", "difficult", "talc"]
        self.img_size = 512
        if self.ready:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.classes = list(meta.get("classes", self.classes))
            self.img_size = int(meta.get("img_size", self.img_size))

    def _session(self) -> ort.InferenceSession:
        if not self.ready:
            raise FileNotFoundError(self.model_path)
        if self.session is None:
            self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        return self.session

    def predict(self, image_path: Path) -> dict:
        if not self.ready:
            return {
                "model_class_name": "unknown",
                "model_display_name": DISPLAY_NAMES["unknown"],
                "model_confidence": None,
                "model_probs": {},
                "model_version": "ore-effb3-onnx-missing",
            }
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        array = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        array = (array - mean) / std
        batch = np.transpose(array, (2, 0, 1))[None].astype(np.float32)
        logits = self._session().run([self.output_name], {self.input_name: batch})[0][0]
        logits = logits.astype(np.float64)
        exp = np.exp(logits - np.max(logits))
        probs_arr = exp / np.sum(exp)
        probs = {label: float(probs_arr[idx]) for idx, label in enumerate(self.classes)}
        class_name = max(probs, key=probs.get)
        return {
            "model_class_name": class_name,
            "model_display_name": DISPLAY_NAMES.get(class_name, class_name),
            "model_confidence": float(probs[class_name]),
            "model_probs": probs,
            "model_version": f"ore-effb3-onnx-{self.img_size}",
        }


class ExpertRuleClassifierService(ClassifierService):
    def __init__(self) -> None:
        self.model = OnnxOreClassifier()

    def classify(self, project_id: str) -> ClassificationResult:
        root = project_dir(project_id)
        result_path = root / "classification.json"
        mask_path = root / "mask_final.png"
        model_inputs = [source_path(project_id), mask_path]
        if self.model.ready:
            model_inputs.extend([self.model.model_path, self.model.meta_path])
        if result_path.exists() and all(result_path.stat().st_mtime_ns >= path.stat().st_mtime_ns for path in model_inputs if path.exists()):
            data = read_json(result_path)
            if "model_class_name" in data and "model_probs" in data:
                return ClassificationResult(project_id=project_id, **data)

        phase = analyze_image(source_path(project_id), mask_path, root)
        model_result = self.model.predict(source_path(project_id))
        stats = phase.stats
        talc_percent = float(stats["talc_percent"])
        model_class = model_result["model_class_name"]
        model_confidence = model_result["model_confidence"]

        if talc_percent > TALC_DECISION_THRESHOLD:
            class_name = "talc"
            display_name = DISPLAY_NAMES[class_name]
            confidence = max(0.9, float(model_confidence or 0.0)) if model_class == "talc" else 0.9
            decision_reason = (
                f"talc_percent={talc_percent:.2f}% > 10%, expert talc rule overrides model "
                f"prediction {model_class}"
            )
            rule_version = "talc_gt_10_overrides_onnx_v1"
        else:
            class_name = model_class
            display_name = DISPLAY_NAMES.get(class_name, class_name)
            confidence = model_confidence
            decision_reason = (
                f"ONNX classifier selected {class_name}; talc_percent={talc_percent:.2f}% <= 10%, "
                "phase masks are shown as auxiliary explanation only"
            )
            rule_version = "onnx_primary_when_talc_le_10_v1"

        data = {
            "class_name": class_name,
            "display_name": display_name,
            "confidence": confidence,
            "probs": model_result["model_probs"],
            "model_version": model_result["model_version"],
            "rule_version": rule_version,
            "decision_reason": decision_reason,
            "phase_stats": stats,
            **model_result,
        }
        write_json(result_path, data)
        return ClassificationResult(project_id=project_id, **data)


class DummyClassifierService(ClassifierService):
    def classify(self, project_id: str) -> ClassificationResult:
        return ClassificationResult(project_id=project_id)
