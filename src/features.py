"""Interpretable image-forensic feature extraction."""

from __future__ import annotations

import io
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.fftpack import dct


FEATURE_NAMES = [
    "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
    "gray_mean", "gray_std", "gray_min", "gray_max", "edge_density",
    "gradient_mean", "gradient_std", "laplacian_variance", "dct_hf_ratio",
    "jpeg_error_mean", "jpeg_error_max", "jpeg_error_p95",
    "block_mean_std", "block_std_mean", "chroma_mean", "chroma_std",
]


def _grayscale(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def _jpeg_error(image: np.ndarray) -> tuple[float, float, float]:
    pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=75)
    buffer.seek(0)
    recompressed = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32)
    error = np.abs(image.astype(np.float32) - recompressed) / 255.0
    return float(error.mean()), float(error.max()), float(np.percentile(error, 95))


def extract_features(image: np.ndarray) -> np.ndarray:
    """Return one fixed-length feature vector for an RGB image."""
    normalized = image.astype(np.float32) / 255.0
    gray = _grayscale(image)
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    gradient = np.sqrt(gx * gx + gy * gy)
    laplacian = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    edge_density = float((gradient > 0.18).mean())
    coefficients = dct(dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")
    hf = np.abs(coefficients[4:, 4:]).mean()
    total = np.abs(coefficients).mean() + 1e-8
    dct_hf_ratio = float(hf / total)
    block_means: list[float] = []
    block_stds: list[float] = []
    height, width = gray.shape
    for y in range(0, height, max(1, height // 4)):
        for x in range(0, width, max(1, width // 4)):
            block = gray[y : y + max(1, height // 4), x : x + max(1, width // 4)]
            block_means.append(float(block.mean()))
            block_stds.append(float(block.std()))
    jpeg_mean, jpeg_max, jpeg_p95 = _jpeg_error(image)
    chroma = normalized[..., 0] - normalized[..., 2]
    values = [
        *[float(normalized[..., channel].mean()) for channel in range(3)],
        *[float(normalized[..., channel].std()) for channel in range(3)],
        float(gray.mean()), float(gray.std()), float(gray.min()), float(gray.max()),
        edge_density, float(gradient.mean()), float(gradient.std()),
        float(laplacian.var()), dct_hf_ratio,
        jpeg_mean, jpeg_max, jpeg_p95,
        float(np.std(block_means)), float(np.mean(block_stds)),
        float(chroma.mean()), float(chroma.std()),
    ]
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError(f"Expected {len(FEATURE_NAMES)} features, got {len(values)}")
    return np.asarray(values, dtype=np.float32)


def extract_frame(frame, progress_every: int = 0) -> tuple[np.ndarray, list[str]]:
    """Extract features from a metadata DataFrame with an image_path column."""
    vectors: list[np.ndarray] = []
    for idx, path in enumerate(frame["image_path"]):
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        vectors.append(extract_features(image))
        if progress_every and (idx + 1) % progress_every == 0:
            print(f"  extracted {idx + 1}/{len(frame)} images")
    return np.vstack(vectors), FEATURE_NAMES.copy()
