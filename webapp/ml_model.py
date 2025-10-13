"""Model loading and inference helpers for the IDS web application."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
_MODEL_LOCK = Lock()
_MODEL_INSTANCE: "ModelService | None" = None


@dataclass
class PredictionResult:
    """Container for structured prediction output."""

    labels: List[str]
    probabilities: List[Dict[str, float]]


class ModelService:
    """Lazily loads the trained pipeline and runs predictions."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._pipeline = joblib.load(self.model_path)
        # Persist feature order to guarantee consistent dataframe slicing.
        if not hasattr(self._pipeline, "feature_names_in_"):
            raise AttributeError(
                "The persisted pipeline is missing `feature_names_in_`. "
                "Retraining with scikit-learn>=1.0 will populate this attribute."
            )
        self.features: List[str] = list(self._pipeline.feature_names_in_)
        self.target_classes: Sequence[str] = getattr(self._pipeline, "classes_", ())

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate incoming frame and align feature columns."""
        df = df.copy()
        missing = [col for col in self.features if col not in df.columns]
        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {', '.join(missing)}"
            )
        # Drop extraneous columns that might appear in exports.
        extra = [col for col in df.columns if col not in self.features]
        if extra:
            df = df.drop(columns=extra)
        # Ensure column ordering.
        df = df[self.features]
        return df

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        """Run the pipeline on the provided dataframe."""
        prepared = self.prepare_dataframe(df)
        labels = self._pipeline.predict(prepared)
        proba = None
        if hasattr(self._pipeline, "predict_proba"):
            proba = self._pipeline.predict_proba(prepared)
        else:
            proba = np.zeros((len(labels), len(self.target_classes)))
            for idx, label in enumerate(labels):
                if label in self.target_classes:
                    class_idx = list(self.target_classes).index(label)
                    proba[idx, class_idx] = 1.0
        probability_dicts = self._to_probability_dicts(proba)
        return PredictionResult(labels=list(map(str, labels)), probabilities=probability_dicts)

    def _to_probability_dicts(self, proba: np.ndarray) -> List[Dict[str, float]]:
        if proba is None or not len(self.target_classes):
            return [{} for _ in range(len(proba) if proba is not None else 0)]
        return [
            {str(cls): float(score) for cls, score in zip(self.target_classes, row)}
            for row in proba
        ]

    def sample_payload(self) -> Dict[str, Iterable[str]]:
        """Return feature order to assist the front-end."""
        return {"features": self.features, "classes": list(map(str, self.target_classes))}

    def as_metadata(self) -> str:
        """Serialize usable metadata for display."""
        metadata = {
            "model_path": str(self.model_path),
            "features": self.features,
            "classes": list(map(str, self.target_classes)),
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
                else Path(__file__).resolve().parents[1] / "models" / "model_random_forest.joblib"
            )
            _MODEL_INSTANCE = ModelService(resolved)
    return _MODEL_INSTANCE
