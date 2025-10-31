"""Inference helpers that wrap the tuned random forest pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Sequence

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

_MODEL_LOCK = Lock()
_MODEL_INSTANCE: "ModelService | None" = None


@dataclass
class PredictionResult:
    """Container for structured prediction output."""

    labels: List[str]
    probabilities: List[Dict[str, float]]


class ModelService:
    """Loads the tuned random forest pipeline and performs predictions."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.pipeline: Pipeline = joblib.load(model_path)
        self.preprocessor = self._extract_preprocessor()
        self.estimator = self._extract_estimator()

        self.cat_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.features: List[str] = self._infer_features()
        self.target_classes: Sequence[str] = [
            str(cls) for cls in getattr(self.estimator, "classes_", [])
        ]

    def _extract_preprocessor(self) -> ColumnTransformer | None:
        if hasattr(self.pipeline, "named_steps"):
            return self.pipeline.named_steps.get("preprocessor")
        return None

    def _extract_estimator(self) -> Any:
        if hasattr(self.pipeline, "named_steps"):
            # Prefer a step named 'model', otherwise take the last step with predict.
            named_steps = self.pipeline.named_steps
            if "model" in named_steps:
                return named_steps["model"]
            for step_name in reversed(list(named_steps.keys())):
                step = named_steps[step_name]
                if hasattr(step, "predict"):
                    return step
        return self.pipeline

    def _infer_features(self) -> List[str]:
        if isinstance(self.preprocessor, ColumnTransformer):
            transformers = getattr(self.preprocessor, "transformers_", None)
            if transformers is None:
                transformers = self.preprocessor.transformers
            for name, _transformer, columns in transformers:
                if columns is None or columns == "drop":
                    continue
                if isinstance(columns, list):
                    if name == "cat":
                        self.cat_columns.extend(str(col) for col in columns)
                    else:
                        self.numeric_columns.extend(str(col) for col in columns)
                else:
                    # Fallback when columns are provided as array-like.
                    try:
                        iter(columns)  # type: ignore[arg-type]
                    except TypeError:
                        columns = [columns]
                    if name == "cat":
                        self.cat_columns.extend(str(col) for col in columns)
                    else:
                        self.numeric_columns.extend(str(col) for col in columns)
            feature_names_in = getattr(self.preprocessor, "feature_names_in_", None)
            if feature_names_in is not None:
                return [str(col) for col in feature_names_in]

        # Fallback to concatenated categorical + numeric columns.
        ordered_features: List[str] = []
        ordered_features.extend(self.cat_columns)
        ordered_features.extend(self.numeric_columns)
        return ordered_features

    def predict(self, rows: List[Dict[str, Any]]) -> PredictionResult:
        if not rows:
            raise ValueError("Input data tidak boleh kosong.")

        frame = pd.DataFrame(rows)
        missing = [feature for feature in self.features if feature not in frame.columns]
        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {', '.join(missing)}"
            )

        frame = frame[self.features]
        labels_raw = self.pipeline.predict(frame)

        if hasattr(self.pipeline, "predict_proba"):
            probabilities_raw = self.pipeline.predict_proba(frame)
        elif hasattr(self.estimator, "predict_proba"):
            probabilities_raw = self.estimator.predict_proba(frame)
        else:
            probabilities_raw = None

        labels = [str(label) for label in labels_raw]
        probabilities: List[Dict[str, float]] = []
        if probabilities_raw is not None:
            for row in probabilities_raw:
                entry = {
                    str(self.target_classes[idx]): float(prob)
                    for idx, prob in enumerate(row)
                }
                probabilities.append(entry)
        else:
            default = 1.0 / max(1, len(self.target_classes))
            probabilities = [
                {str(target): default for target in self.target_classes}
                for _ in labels
            ]

        return PredictionResult(labels=labels, probabilities=probabilities)

    def sample_payload(self) -> Dict[str, Iterable[str]]:
        return {"features": self.features, "classes": list(self.target_classes)}

    def as_metadata(self) -> str:
        metadata = {
            "model_path": str(self.model_path),
            "features": self.features,
            "classes": list(self.target_classes),
            "cat_columns": self.cat_columns,
            "numeric_columns": self.numeric_columns,
            "estimator": type(self.estimator).__name__,
        }
        return json.dumps(metadata, indent=2)


def get_model_service(model_path: Path | None = None) -> ModelService:
    """Return the singleton model service."""
    global _MODEL_INSTANCE
    with _MODEL_LOCK:
        if _MODEL_INSTANCE is None:
            resolved = (
                model_path
                if model_path is not None
                else Path(__file__).resolve().parents[1]
                / "models"
                / "tuned_random_forest_model.joblib"
            )
            _MODEL_INSTANCE = ModelService(resolved)
    return _MODEL_INSTANCE
