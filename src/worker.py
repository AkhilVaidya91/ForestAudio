from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audio import audio_to_tensor, fallback_multiclass_prediction, load_class_names, load_model, predict_with_model
from .config import AppConfig
from .storage import LocalStore


@dataclass(slots=True)
class CloudInferenceBackend:
    config: AppConfig
    class_names: list[str] = field(init=False)
    model: Any | None = field(init=False)

    def __post_init__(self) -> None:
        self.class_names = load_class_names(self.config)
        self.model = load_model(self.config.model_path, len(self.class_names))

    def predict(self, audio_path: Path) -> tuple[str, float, dict[str, float]]:
        if self.model is None:
            class_index, confidence, probabilities = fallback_multiclass_prediction(audio_path, self.class_names)
        else:
            tensor = audio_to_tensor(audio_path)
            class_index, confidence, probabilities = predict_with_model(self.model, tensor)
        predicted_class = self.class_names[class_index] if class_index < len(self.class_names) else f"class_{class_index}"
        probability_map = {
            self.class_names[index] if index < len(self.class_names) else f"class_{index}": round(float(probability), 4)
            for index, probability in enumerate(probabilities)
        }
        return predicted_class, confidence, probability_map


class QueueWorker:
    def __init__(self, config: AppConfig, store: LocalStore) -> None:
        self.config = config
        self.store = store
        self.backend = CloudInferenceBackend(config)

    def run_forever(self, poll_interval: float = 2.0) -> None:
        while True:
            job = self.store.claim_next_job()
            if job is None:
                time.sleep(poll_interval)
                continue
            upload = self.store.get_upload(job.upload_id)
            if upload is None:
                continue
            predicted_class, confidence, probability_map = self.backend.predict(Path(upload.path))
            self.store.record_alert(
                upload_id=job.upload_id,
                node_id=upload.node_id,
                lat=upload.lat,
                lon=upload.lon,
                edge_confidence=upload.edge_confidence,
                predicted_class=predicted_class,
                cloud_confidence=confidence,
                probabilities=probability_map,
                inference_backend="pytorch" if self.backend.model is not None else "fallback",
                created_at=time.time(),
            )
            self.store.mark_upload_processed(job.upload_id)


def run_worker(config: AppConfig, store: LocalStore) -> None:
    QueueWorker(config, store).run_forever()
