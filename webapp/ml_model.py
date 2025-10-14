"""Pure-Python inference helpers for the IDS web application."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Sequence

_MODEL_LOCK = Lock()
_MODEL_INSTANCE: "ModelService | None" = None


@dataclass
class PredictionResult:
    """Container for structured prediction output."""

    labels: List[str]
    probabilities: List[Dict[str, float]]


class CompiledTreeModel:
    """Runtime for a decision tree classifier compiled from scikit-learn."""

    def __init__(self, tree_config: Dict[str, Any]) -> None:
        self.classes: List[str] = [str(cls) for cls in tree_config["classes"]]
        self.children_left: List[int] = tree_config["children_left"]
        self.children_right: List[int] = tree_config["children_right"]
        self.features: List[int] = tree_config["feature"]
        self.thresholds: List[float] = tree_config["threshold"]
        # value[node] is a list of class counts.
        self.values: List[List[float]] = [
            [float(v) for v in row] for row in tree_config["value"]
        ]

    def predict(self, vector: List[float]) -> tuple[str, List[float]]:
        node = 0
        while self.children_left[node] != -1:
            feature_idx = self.features[node]
            threshold = self.thresholds[node]
            if vector[feature_idx] <= threshold:
                node = self.children_left[node]
            else:
                node = self.children_right[node]
        counts = self.values[node]
        total = sum(counts)
        if total > 0.0:
            probabilities = [count / total for count in counts]
        else:
            probabilities = [1.0 / len(counts) for _ in counts]
        best_idx = max(range(len(probabilities)), key=probabilities.__getitem__)
        return self.classes[best_idx], probabilities


class ModelService:
    """Loads the compiled model configuration and performs predictions."""

    def __init__(self, model_path: Path) -> None:
        with model_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
        self.model_path = model_path
        self.features: List[str] = list(config["features"])

        self.cat_columns: List[str] = list(config["cat_columns"])
        self.cat_categories: List[List[str]] = [
            [str(cat) for cat in cats] for cats in config["cat_categories"]
        ]
        self.cat_offsets: List[int] = []
        offset = 0
        for cats in self.cat_categories:
            self.cat_offsets.append(offset)
            offset += len(cats)
        self.cat_dimension = offset

        self.numeric_columns: List[str] = list(config["numeric_columns"])
        self.numeric_mean: List[float] = [float(v) for v in config["numeric_mean"]]
        self.numeric_scale: List[float] = [float(v) for v in config["numeric_scale"]]

        self.numeric_dimension = len(self.numeric_columns)
        self.total_dimension = self.cat_dimension + self.numeric_dimension

        self.tree = CompiledTreeModel(config["tree"])
        self.target_classes: Sequence[str] = self.tree.classes

    def _prepare_row(self, row: Dict[str, Any]) -> List[float]:
        missing = [feature for feature in self.features if feature not in row]
        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {', '.join(missing)}"
            )

        vector = [0.0] * self.total_dimension

        # One-hot encode categorical features.
        for col_idx, column in enumerate(self.cat_columns):
            categories = self.cat_categories[col_idx]
            offset = self.cat_offsets[col_idx]
            value = row.get(column)
            if value is None:
                continue
            try:
                str_value = str(value)
                cat_index = categories.index(str_value)
            except ValueError:
                continue
            vector[offset + cat_index] = 1.0

        # Scale numeric features.
        for idx, column in enumerate(self.numeric_columns):
            raw_value = row.get(column, 0)
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                numeric_value = 0.0
            mean = self.numeric_mean[idx]
            scale = self.numeric_scale[idx] if self.numeric_scale[idx] else 1.0
            vector[self.cat_dimension + idx] = (numeric_value - mean) / scale

        return vector

    def _prepare_samples(self, rows: List[Dict[str, Any]]) -> List[List[float]]:
        return [self._prepare_row(row) for row in rows]

    def predict(self, rows: List[Dict[str, Any]]) -> PredictionResult:
        prepared_vectors = self._prepare_samples(rows)
        labels: List[str] = []
        probabilities: List[Dict[str, float]] = []
        for vector in prepared_vectors:
            label, probs = self.tree.predict(vector)
            labels.append(label)
            probabilities.append(
                {self.tree.classes[idx]: prob for idx, prob in enumerate(probs)}
            )
        return PredictionResult(labels=labels, probabilities=probabilities)

    def sample_payload(self) -> Dict[str, Iterable[str]]:
        return {"features": self.features, "classes": list(map(str, self.target_classes))}

    def as_metadata(self) -> str:
        metadata = {
            "model_path": str(self.model_path),
            "features": self.features,
            "classes": list(map(str, self.target_classes)),
            "cat_columns": self.cat_columns,
            "numeric_columns": self.numeric_columns,
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
                else Path(__file__).resolve().parents[1] / "models" / "model_compiled.json"
            )
            _MODEL_INSTANCE = ModelService(resolved)
    return _MODEL_INSTANCE
