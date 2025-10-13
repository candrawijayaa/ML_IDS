"""Flask entrypoint exposing IDS predictions via web UI and JSON API."""
from __future__ import annotations

import os
import tempfile
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.datastructures import FileStorage

from .ml_model import PredictionResult, get_model_service
from pcap_kdd_extractor import ConnRecord, process_pcap

MAX_PREVIEW_ROWS = 50


def create_app(model_path: str | Path | None = None) -> Flask:
    app = Flask(__name__)
    model_service = get_model_service(Path(model_path) if model_path else None)

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            features=model_service.features,
            classes=model_service.target_classes,
            metadata=model_service.as_metadata(),
        )

    @app.post("/predict")
    def predict_form() -> str:
        file = request.files.get("file")
        if not _is_valid_file(file):
            return render_template(
                "index.html",
                features=model_service.features,
                classes=model_service.target_classes,
                metadata=model_service.as_metadata(),
                errors=["Mohon unggah berkas PCAP yang valid."],
            ), HTTPStatus.BAD_REQUEST
        try:
            dataframe = _read_uploaded_pcap(file)
            prediction = model_service.predict(dataframe)
            preview = _results_to_rows(dataframe, prediction, limit=MAX_PREVIEW_ROWS)
        except ValueError as exc:
            return render_template(
                "index.html",
                features=model_service.features,
                classes=model_service.target_classes,
                metadata=model_service.as_metadata(),
                errors=[str(exc)],
            ), HTTPStatus.BAD_REQUEST
        except Exception as exc:  # pragma: no cover - defensive handling
            return render_template(
                "index.html",
                features=model_service.features,
                classes=model_service.target_classes,
                metadata=model_service.as_metadata(),
                errors=[f"Terjadi kesalahan saat memproses berkas: {exc}"],
            ), HTTPStatus.INTERNAL_SERVER_ERROR
        return render_template(
            "index.html",
            features=model_service.features,
            classes=model_service.target_classes,
            metadata=model_service.as_metadata(),
            preview_rows=preview,
            total_rows=len(prediction.labels),
        )

    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True)
        if not payload or "samples" not in payload:
            return (
                jsonify(
                    {
                        "error": "JSON payload harus menyertakan kunci 'samples' berisi daftar objek fitur.",
                        "expected_features": model_service.features,
                    }
                ),
                HTTPStatus.BAD_REQUEST,
            )
        samples = payload["samples"]
        if not isinstance(samples, list):
            return (
                jsonify({"error": "'samples' harus berupa list objek fitur."}),
                HTTPStatus.BAD_REQUEST,
            )
        dataframe = pd.DataFrame(samples)
        try:
            prediction = model_service.predict(dataframe)
        except ValueError as exc:
            return (
                jsonify(
                    {
                        "error": str(exc),
                        "expected_features": model_service.features,
                    }
                ),
                HTTPStatus.BAD_REQUEST,
            )
        response = {
            "predictions": prediction.labels,
            "probabilities": prediction.probabilities,
            "classes": list(map(str, model_service.target_classes)),
        }
        return jsonify(response)

    @app.get("/health")
    def healthcheck():
        return {"status": "ok", "model_loaded": bool(model_service.features)}

    return app


def _is_valid_file(file: FileStorage | None) -> bool:
    return bool(file and file.filename and file.filename.lower().endswith(".pcap"))


def _read_uploaded_pcap(file: FileStorage) -> pd.DataFrame:
    data = file.read()
    file.seek(0)
    if not data:
        raise ValueError("Berkas PCAP kosong.")

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        connections = process_pcap(tmp_path)
    except Exception as exc:
        raise ValueError(f"Berkas PCAP tidak dapat diproses: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not connections:
        raise ValueError("Tidak ditemukan koneksi yang dapat diekstrak dari PCAP.")

    rows = [_conn_record_to_row(conn) for conn in connections]
    dataframe = pd.DataFrame(rows)
    return dataframe


def _conn_record_to_row(conn: ConnRecord) -> Dict[str, Any]:
    duration = max(0.0, conn.end - conn.start)
    return {
        "duration": duration,
        "protocol_type": conn.proto,
        "service": conn.service,
        "flag": conn.flag,
        "src_bytes": conn.src_bytes,
        "dst_bytes": conn.dst_bytes,
        "land": conn.land,
        "wrong_fragment": conn.wrong_fragment,
        "urgent": conn.urgent,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 0,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": conn.count,
        "srv_count": conn.srv_count,
        "serror_rate": conn.serror_rate,
        "srv_serror_rate": conn.srv_serror_rate,
        "rerror_rate": conn.rerror_rate,
        "srv_rerror_rate": conn.srv_rerror_rate,
        "same_srv_rate": conn.same_srv_rate,
        "diff_srv_rate": conn.diff_srv_rate,
        "srv_diff_host_rate": conn.srv_diff_host_rate,
        "dst_host_count": conn.dst_host_count,
        "dst_host_srv_count": conn.dst_host_srv_count,
        "dst_host_same_srv_rate": conn.dst_host_same_srv_rate,
        "dst_host_diff_srv_rate": conn.dst_host_diff_srv_rate,
        "dst_host_same_src_port_rate": conn.dst_host_same_src_port_rate,
        "dst_host_srv_diff_host_rate": conn.dst_host_srv_diff_host_rate,
        "dst_host_serror_rate": conn.dst_host_serror_rate,
        "dst_host_srv_serror_rate": conn.dst_host_srv_serror_rate,
        "dst_host_rerror_rate": conn.dst_host_rerror_rate,
        "dst_host_srv_rerror_rate": conn.dst_host_srv_rerror_rate,
    }


def _results_to_rows(
    df: pd.DataFrame, prediction: PredictionResult, limit: int | None = None
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    preview_len = min(len(df), limit) if limit else len(df)
    for idx in range(preview_len):
        label = prediction.labels[idx]
        row = {col: df.iloc[idx][col] for col in df.columns}
        row["predicted_label"] = label
        proba = prediction.probabilities[idx]
        if proba:
            top_class = max(proba, key=proba.get)
            row["confidence"] = round(float(proba[top_class]) * 100, 2)
        rows.append(row)
    return rows


app = create_app()

if __name__ == "__main__":
    port_env = os.environ.get("PORT") or os.environ.get("FLASK_RUN_PORT")
    try:
        port = int(port_env) if port_env else 5000
    except ValueError:
        port = 5000
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=True)
