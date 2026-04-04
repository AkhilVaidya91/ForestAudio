from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import time


@dataclass(slots=True)
class NodeState:
    node_id: str
    lat: float
    lon: float
    status: str = "active"
    last_heartbeat: float = 0.0
    last_prediction: int | None = None
    last_confidence: float | None = None
    last_audio: str | None = None


@dataclass(slots=True)
class SampleAssignment:
    node_id: str
    audio_path: Path
    label_id: int
    class_name: str


@dataclass(slots=True)
class EdgeDecision:
    node_id: str
    audio_path: Path
    label_id: int
    class_name: str
    binary_prediction: int
    confidence: float
    edge_probability: float
    created_at: float = field(default_factory=time)


@dataclass(slots=True)
class CloudAlert:
    upload_id: str
    node_id: str
    lat: float
    lon: float
    filename: str
    predicted_class: str
    cloud_confidence: float
    edge_confidence: float | None
    inference_backend: str
    probabilities: dict[str, float]
    created_at: float = field(default_factory=time)
