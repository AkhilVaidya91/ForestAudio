from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from .audio import infer_label_id
from .config import AppConfig
from .models import SampleAssignment


@dataclass(slots=True)
class EnvironmentSampler:
    config: AppConfig
    rng: random.Random
    audio_files: list[Path] = field(init=False)
    anomaly_pool: list[Path] = field(init=False)
    normal_pool: list[Path] = field(init=False)

    def __post_init__(self) -> None:
        self.audio_files = sorted(self.config.audio_dir.glob("*.wav"))
        self.anomaly_pool = [path for path in self.audio_files if infer_label_id(path) in self.config.target_class_labels]
        self.normal_pool = [path for path in self.audio_files if infer_label_id(path) not in self.config.target_class_labels]
        if not self.audio_files:
            raise FileNotFoundError(f"No .wav files found in {self.config.audio_dir}")

    def next_batch(self, node_ids: list[str]) -> list[SampleAssignment]:
        anomaly_count = sum(1 for _ in node_ids if self.rng.random() < self.config.anomaly_rate)
        anomaly_count = min(anomaly_count, len(self.anomaly_pool))
        normal_count = len(node_ids) - anomaly_count

        anomaly_samples = self.rng.sample(self.anomaly_pool, anomaly_count) if anomaly_count else []
        normal_samples = self.rng.sample(self.normal_pool, normal_count) if normal_count else []

        assignments: list[SampleAssignment] = []
        anomaly_index = 0
        normal_index = 0
        for node_id in node_ids:
            if anomaly_index < len(anomaly_samples):
                audio_path = anomaly_samples[anomaly_index]
                anomaly_index += 1
            else:
                audio_path = normal_samples[normal_index]
                normal_index += 1
            label_id = infer_label_id(audio_path)
            assignments.append(SampleAssignment(node_id=node_id, audio_path=audio_path, label_id=label_id, class_name=f"class_{label_id}"))
        return assignments
