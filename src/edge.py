from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .audio import audio_to_tensor, edge_probability_for_class_names, fallback_multiclass_prediction, load_class_names, load_model, predict_with_model
from .config import AppConfig
from .models import EdgeDecision, SampleAssignment


@dataclass(slots=True)
class EdgeModelEngine:
    config: AppConfig
    class_names: list[str] = field(init=False)
    model: Any | None = field(init=False)

    def __post_init__(self) -> None:
        self.class_names = load_class_names(self.config)
        self.model = load_model(self.config.model_path, len(self.class_names))

    def predict(self, assignment: SampleAssignment) -> EdgeDecision:
        if self.model is None:
            class_index, confidence, probabilities = fallback_multiclass_prediction(assignment.audio_path, self.class_names)
        else:
            tensor = audio_to_tensor(assignment.audio_path)
            class_index, confidence, probabilities = predict_with_model(self.model, tensor)
        class_name = self.class_names[class_index] if class_index < len(self.class_names) else f"class_{class_index}"
        target_probability = edge_probability_for_class_names(probabilities, self.class_names, self.config.target_class_labels)
        binary_prediction = 1 if target_probability >= self.config.edge_threshold else 0
        return EdgeDecision(
            node_id=assignment.node_id,
            audio_path=assignment.audio_path,
            label_id=assignment.label_id,
            class_name=class_name,
            binary_prediction=binary_prediction,
            confidence=confidence,
            edge_probability=target_probability,
        )
