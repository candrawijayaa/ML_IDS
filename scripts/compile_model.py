#!/usr/bin/env python3
"""Konversi pipeline/joblib scikit-learn menjadi konfigurasi JSON untuk webapp ML_IDS."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


class CompilationError(RuntimeError):
    """Kesalahan khusus saat mengubah model ke format JSON."""


@dataclass
class PreprocessorMetadata:
    features: List[str]
    cat_columns: List[str]
    cat_categories: List[List[str]]
    numeric_columns: List[str]
    numeric_mean: List[float]
    numeric_scale: List[float]


def _unwrap_transformer(transformer: Any) -> Any:
    """Jika transformer dibungkus Pipeline, ambil estimator terakhirnya."""
    if isinstance(transformer, Pipeline):
        return transformer.steps[-1][1]
    return transformer


def _extract_preprocessor_metadata(
    model: Pipeline | Any,
) -> Tuple[PreprocessorMetadata, Any]:
    """Ambil metadata preprocessing + estimator akhir dari pipeline joblib."""
    if isinstance(model, Pipeline):
        steps = dict(model.named_steps)
        preprocessor = steps.get("preprocessor")
        estimator = steps.get("model") or steps.get("classifier")
    else:
        preprocessor = None
        estimator = model
    if preprocessor is None:
        raise CompilationError("Pipeline harus memiliki langkah 'preprocessor'.")
    if estimator is None:
        raise CompilationError("Pipeline tidak memiliki langkah estimator 'model'/'classifier'.")
    if not isinstance(preprocessor, ColumnTransformer):
        raise CompilationError("Langkah preprocessor harus ColumnTransformer.")

    cat_columns: List[str] = []
    cat_categories: List[List[str]] = []
    numeric_columns: List[str] = []
    numeric_mean: List[float] = []
    numeric_scale: List[float] = []

    for name, transformer, columns in preprocessor.transformers_:
        if transformer == "drop":
            continue
        resolved = _unwrap_transformer(transformer)
        cols = [str(col) for col in columns]
        if isinstance(resolved, OneHotEncoder):
            cat_columns = cols
            cat_categories = [
                [str(value) for value in cats] for cats in resolved.categories_
            ]
        elif isinstance(resolved, StandardScaler):
            numeric_columns = cols
            numeric_mean = [float(val) for val in resolved.mean_]
            numeric_scale = [float(val) for val in resolved.scale_]

    if not cat_columns or not cat_categories:
        raise CompilationError("Tidak menemukan transformer kategorikal OneHotEncoder.")
    if not numeric_columns or not numeric_mean:
        raise CompilationError("Tidak menemukan transformer numerik StandardScaler.")

    features: Iterable[str]
    if hasattr(model, "feature_names_in_"):
        features = model.feature_names_in_
    elif hasattr(preprocessor, "feature_names_in_"):
        features = preprocessor.feature_names_in_
    else:
        raise CompilationError("Pipeline tidak menyimpan urutan fitur input.")

    metadata = PreprocessorMetadata(
        features=[str(name) for name in features],
        cat_columns=cat_columns,
        cat_categories=cat_categories,
        numeric_columns=numeric_columns,
        numeric_mean=numeric_mean,
        numeric_scale=numeric_scale,
    )
    return metadata, estimator


def _serialize_tree(estimator: DecisionTreeClassifier) -> Dict[str, Any]:
    """Ubah DecisionTreeClassifier menjadi struktur JSON sederhana."""
    tree = estimator.tree_
    values = tree.value
    if values.ndim == 3:
        values = values[:, 0, :]
    return {
        "classes": [str(label) for label in estimator.classes_],
        "children_left": tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.astype(float).tolist(),
        "value": np.array(values, dtype=float).tolist(),
    }


def compile_model(joblib_path: Path, output_path: Path) -> None:
    """Proses utama untuk mengubah file joblib menjadi konfigurasi JSON."""
    pipeline = joblib.load(joblib_path)
    metadata, estimator = _extract_preprocessor_metadata(pipeline)

    config: Dict[str, Any] = {
        "features": metadata.features,
        "cat_columns": metadata.cat_columns,
        "cat_categories": metadata.cat_categories,
        "numeric_columns": metadata.numeric_columns,
        "numeric_mean": metadata.numeric_mean,
        "numeric_scale": metadata.numeric_scale,
    }

    if isinstance(estimator, RandomForestClassifier):
        trees = [_serialize_tree(tree) for tree in estimator.estimators_]
        config["classes"] = [str(label) for label in estimator.classes_]
        config["n_estimators"] = len(trees)
        config["forest"] = trees
        if trees:
            config["tree"] = trees[0]
    elif isinstance(estimator, DecisionTreeClassifier):
        config["classes"] = [str(label) for label in estimator.classes_]
        config["n_estimators"] = 1
        config["tree"] = _serialize_tree(estimator)
    else:
        raise CompilationError(
            f"Tipe estimator {type(estimator)} belum didukung. Gunakan RandomForestClassifier atau DecisionTreeClassifier."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kompilasi model joblib scikit-learn menjadi JSON untuk runtime ML_IDS."
    )
    parser.add_argument("joblib_path", type=Path, help="Path ke model .joblib hasil pelatihan.")
    parser.add_argument(
        "output_path", type=Path, help="Path keluaran file JSON yang akan dibuat."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compile_model(args.joblib_path, args.output_path)


if __name__ == "__main__":
    main()
