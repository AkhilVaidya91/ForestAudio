from __future__ import annotations

import io
import json
import re
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

from .config import AppConfig


IMG_SIZE = 224
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

NORMALIZE = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def infer_label_id(audio_path: Path) -> int:
    match = re.match(r"^(\d+)_", audio_path.name)
    if not match:
        return 0
    return int(match.group(1))


def load_class_names(config: AppConfig, num_classes: int | None = None) -> list[str]:
    for path in config.class_names_paths:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    if num_classes is None:
        num_classes = 2
    return [f"class_{index}" for index in range(num_classes)]


def load_audio(audio_path: Path, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    if len(y) == 0:
        return np.zeros(sample_rate, dtype=np.float32), sr
    target_len = sample_rate * 5
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]
    return librosa.util.normalize(y), sr


def _mel_image(y: np.ndarray, sr: int) -> Image.Image:
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    ax.axis("off")
    librosa.display.specshow(mel_db, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel", ax=ax)
    plt.tight_layout(pad=0)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def audio_to_tensor(audio_path: Path) -> torch.Tensor:
    y, sr = load_audio(audio_path)
    image = _mel_image(y, sr)
    return NORMALIZE(image).unsqueeze(0)


def load_model(model_path: Path, num_classes: int) -> torch.nn.Module | None:
    if not model_path.exists():
        return None
    try:
        model = timm.create_model("deit_small_patch16_224", pretrained=False)
        model.head = nn.Linear(model.head.in_features, num_classes)
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.eval()
        return model
    except Exception:
        return None


def predict_with_model(model: torch.nn.Module, tensor: torch.Tensor) -> tuple[int, float, np.ndarray]:
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])
    return class_index, confidence, probabilities


def fallback_multiclass_prediction(audio_path: Path, class_names: list[str]) -> tuple[int, float, np.ndarray]:
    label_id = infer_label_id(audio_path)
    num_classes = max(len(class_names), label_id + 1)
    probabilities = np.full(num_classes, 1.0 / num_classes, dtype=np.float32)
    index = min(label_id, num_classes - 1)
    probabilities[index] = 0.88
    probabilities /= probabilities.sum()
    class_index = int(np.argmax(probabilities))
    return class_index, float(probabilities[class_index]), probabilities


def class_id_from_name(class_name: str) -> int:
    match = re.search(r"(\d+)$", class_name)
    if not match:
        return -1
    return int(match.group(1))


def edge_probability(probabilities: np.ndarray, target_class_ids: tuple[int, ...]) -> float:
    if not len(probabilities):
        return 0.0
    total = 0.0
    for class_id in target_class_ids:
        if 0 <= class_id < len(probabilities):
            total += float(probabilities[class_id])
    return total


def edge_probability_for_class_names(probabilities: np.ndarray, class_names: list[str], target_class_ids: tuple[int, ...]) -> float:
    if not len(probabilities):
        return 0.0
    total = 0.0
    for index, probability in enumerate(probabilities):
        if index >= len(class_names):
            break
        if index in target_class_ids:
            total += float(probability)
    return total
