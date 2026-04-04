from __future__ import annotations

import time

from flask import Flask, request, send_file

from .config import AppConfig
from .storage import LocalStore


def create_app(config: AppConfig | None = None, store: LocalStore | None = None) -> Flask:
    config = config or AppConfig()
    store = store or LocalStore(config.db_path)
    config.upload_dir.mkdir(parents=True, exist_ok=True)
    index_path = config.project_root / "index.html"

    app = Flask(__name__)

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.get("/")
    def dashboard() -> tuple[object, int] | object:
        if index_path.exists():
            return send_file(index_path)
        return {"error": "index.html not found"}, 404

    @app.route("/health", methods=["OPTIONS"])
    @app.route("/nodes", methods=["OPTIONS"])
    @app.route("/alerts", methods=["OPTIONS"])
    @app.route("/uploads", methods=["OPTIONS"])
    @app.route("/heartbeat", methods=["OPTIONS"])
    @app.route("/ingest", methods=["OPTIONS"])
    def options_handler() -> tuple[dict[str, str], int]:
        return {}, 204

    @app.get("/health")
    def health() -> tuple[dict[str, object], int]:
        return {"status": "ok", "queue_size": store.pending_jobs_count()}, 200

    @app.post("/heartbeat")
    def heartbeat() -> tuple[dict[str, object], int]:
        payload = request.get_json(force=True, silent=True) or {}
        node_id = str(payload.get("node_id", "unknown-node"))
        lat = float(payload.get("lat", config.center_lat))
        lon = float(payload.get("lon", config.center_lon))
        status = str(payload.get("status", "active"))
        timestamp = float(payload.get("timestamp", time.time()))
        store.update_heartbeat(node_id=node_id, lat=lat, lon=lon, status=status, timestamp=timestamp)
        return {"status": "ok", "node_id": node_id}, 200

    @app.get("/nodes")
    def nodes() -> tuple[dict[str, object], int]:
        rows = store.list_nodes()
        return {
            "nodes": [
                {
                    "node_id": row.node_id,
                    "lat": row.lat,
                    "lon": row.lon,
                    "status": row.status,
                    "last_heartbeat": row.last_heartbeat,
                    "last_prediction": row.last_prediction,
                    "last_confidence": row.last_confidence,
                    "last_audio": row.last_audio,
                }
                for row in rows
            ]
        }, 200

    @app.get("/alerts")
    def alerts() -> tuple[dict[str, object], int]:
        rows = store.list_recent_alerts()
        return {
            "alerts": [
                {
                    "upload_id": row.upload_id,
                    "node_id": row.node_id,
                    "lat": row.lat,
                    "lon": row.lon,
                    "edge_confidence": row.edge_confidence,
                    "predicted_class": row.predicted_class,
                    "cloud_confidence": row.cloud_confidence,
                    "probabilities": row.probabilities_json,
                    "created_at": row.created_at,
                    "backend": row.inference_backend,
                }
                for row in rows
            ]
        }, 200

    @app.get("/uploads")
    def uploads() -> tuple[dict[str, object], int]:
        rows = store.list_uploads()
        return {
            "uploads": [
                {
                    "upload_id": row.upload_id,
                    "node_id": row.node_id,
                    "lat": row.lat,
                    "lon": row.lon,
                    "edge_confidence": row.edge_confidence,
                    "filename": row.filename,
                    "created_at": row.created_at,
                    "status": row.status,
                }
                for row in rows
            ]
        }, 200

    @app.post("/ingest")
    def ingest() -> tuple[dict[str, object], int]:
        upload = request.files.get("file")
        if upload is None:
            return {"error": "file field is required"}, 400
        node_id = request.form.get("node_id", "unknown-node")
        lat = float(request.form.get("lat", config.center_lat))
        lon = float(request.form.get("lon", config.center_lon))
        edge_confidence = request.form.get("edge_confidence")
        edge_confidence_value = float(edge_confidence) if edge_confidence not in {None, ""} else None
        source_name = request.form.get("filename") or upload.filename or f"{node_id}.wav"
        upload_id = request.form.get("upload_id") or f"{node_id}-{int(time.time() * 1000)}"
        saved_path = config.upload_dir / f"{upload_id}.wav"
        upload.save(saved_path)
        store.store_upload(
            upload_id=upload_id,
            path=saved_path,
            node_id=node_id,
            lat=lat,
            lon=lon,
            edge_confidence=edge_confidence_value,
            filename=source_name,
            created_at=time.time(),
        )
        store.enqueue_job(upload_id)
        store.upsert_node(node_id=node_id, lat=lat, lon=lon, status="active")
        return {"status": "queued", "upload_id": upload_id, "node_id": node_id}, 202

    return app
