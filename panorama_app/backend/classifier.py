from __future__ import annotations

from .schemas import ClassificationResult


class ClassifierService:
    def classify(self, project_id: str) -> ClassificationResult:
        raise NotImplementedError


class DummyClassifierService(ClassifierService):
    def classify(self, project_id: str) -> ClassificationResult:
        return ClassificationResult(project_id=project_id)

