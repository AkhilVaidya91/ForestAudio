from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class AppConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    audio_dir: Path = field(init=False)
    upload_dir: Path = field(init=False)
    db_path: Path = field(init=False)
    model_path: Path = field(init=False)
    class_names_paths: tuple[Path, ...] = field(init=False)
    cloud_host: str = field(default_factory=lambda: os.getenv("FORESTAUDIO_CLOUD_HOST", "127.0.0.1"))
    cloud_port: int = field(default_factory=lambda: _env_int("FORESTAUDIO_CLOUD_PORT", 5001))
    dashboard_port: int = field(default_factory=lambda: _env_int("FORESTAUDIO_DASHBOARD_PORT", 8501))
    sample_interval_seconds: float = field(default_factory=lambda: _env_float("FORESTAUDIO_SAMPLE_INTERVAL_SECONDS", 20.0))
    heartbeat_interval_seconds: float = field(default_factory=lambda: _env_float("FORESTAUDIO_HEARTBEAT_INTERVAL_SECONDS", 60.0))
    anomaly_rate: float = field(default_factory=lambda: _env_float("FORESTAUDIO_ANOMALY_RATE", 0.3))
    target_class_labels: tuple[int, ...] = field(default_factory=lambda: tuple(int(value) for value in os.getenv("FORESTAUDIO_TARGET_CLASSES", "1").split(",") if value))
    edge_threshold: float = field(default_factory=lambda: _env_float("FORESTAUDIO_EDGE_THRESHOLD", 0.4))
    center_lat: float = field(default_factory=lambda: _env_float("FORESTAUDIO_CENTER_LAT", 29.5521))
    center_lon: float = field(default_factory=lambda: _env_float("FORESTAUDIO_CENTER_LON", 78.8832))
    map_span: float = field(default_factory=lambda: _env_float("FORESTAUDIO_MAP_SPAN", 0.99))
    max_nodes: int = field(default_factory=lambda: _env_int("FORESTAUDIO_MAX_NODES", 1000))
    enable_triton: bool = field(default_factory=lambda: _env_bool("FORESTAUDIO_ENABLE_TRITON", False))
    triton_url: str = field(default_factory=lambda: os.getenv("FORESTAUDIO_TRITON_URL", "http://127.0.0.1:8000"))
    triton_model_name: str = field(default_factory=lambda: os.getenv("FORESTAUDIO_TRITON_MODEL_NAME", "forest_audio"))

    def __post_init__(self) -> None:
        self.audio_dir = self.project_root / "data" / "audio"
        self.upload_dir = self.project_root / "data" / "uploads"
        self.db_path = self.project_root / "data" / "forest_audio.sqlite3"
        self.model_path = self.project_root / "models" / "deit_mel_spectrogram.pth"
        self.class_names_paths = (
            self.project_root / "class_names.json",
            self.project_root / "reference_code" / "class_names.json",
        )


def get_config() -> AppConfig:
    return AppConfig()
