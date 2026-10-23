"""Download CIFAR-10 and build a reproducible, source-safe tamper benchmark."""

from __future__ import annotations

import io
import pickle
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter


CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def download_and_extract_cifar(root: Path, force: bool = False) -> Path:
    """Download CIFAR-10 from the official source and return its extracted path."""
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "cifar-10-python.tar.gz"
    extracted = root / "cifar-10-batches-py"
    if extracted.exists() and not force:
        return extracted
    if force or not archive.exists():
        print(f"Downloading CIFAR-10 from {CIFAR_URL} ...")
        urllib.request.urlretrieve(CIFAR_URL, archive)
    print("Extracting CIFAR-10 ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    if not extracted.exists():
        raise FileNotFoundError(f"Expected extracted dataset at {extracted}")
    return extracted


def load_cifar_train(extracted: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 train images as N,H,W,C uint8 arrays."""
    images: list[np.ndarray] = []
    labels: list[int] = []
    for batch_idx in range(1, 6):
        path = extracted / f"data_batch_{batch_idx}"
        with path.open("rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        data = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
        images.append(data.transpose(0, 2, 3, 1))
        labels.extend(int(value) for value in batch[b"labels"])
    return np.concatenate(images, axis=0), np.asarray(labels, dtype=np.int64)


def load_fast_fallback() -> tuple[np.ndarray, np.ndarray]:
    """Load a bundled scikit-learn image set when a large download is incomplete.

    The digits set is a transparent, credential-free emergency pilot source. It is
    upscaled and converted to RGB so that the exact same tamper pipeline can run.
    It must not be described as a replacement for CASIA/CoMoFoD in the final study.
    """
    from sklearn.datasets import load_digits

    digits = load_digits()
    grayscale = digits.images.astype(np.float32)
    grayscale = ((grayscale - grayscale.min()) / (grayscale.max() - grayscale.min()) * 255).astype(np.uint8)
    rgb = np.repeat(grayscale[..., None], 3, axis=-1)
    expanded = np.stack(
        [np.asarray(Image.fromarray(image).resize((32, 32), Image.Resampling.NEAREST)) for image in rgb],
        axis=0,
    )
    return expanded, digits.target.astype(np.int64)


def _copy_move(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    patch_w = rng.randint(max(5, width // 5), max(6, width // 3))
    patch_h = rng.randint(max(5, height // 5), max(6, height // 3))
    sx = rng.randint(0, width - patch_w)
    sy = rng.randint(0, height - patch_h)
    patch = image.crop((sx, sy, sx + patch_w, sy + patch_h))
    dx = rng.randint(0, width - patch_w)
    dy = rng.randint(0, height - patch_h)
    if abs(dx - sx) + abs(dy - sy) < patch_w // 2:
        dx = max(0, width - patch_w - dx)
    image = image.copy()
    image.paste(patch, (dx, dy))
    return image


def _splicing(image: Image.Image, donor: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    patch_w = rng.randint(max(5, width // 5), max(6, width // 3))
    patch_h = rng.randint(max(5, height // 5), max(6, height // 3))
    sx = rng.randint(0, width - patch_w)
    sy = rng.randint(0, height - patch_h)
    dx = rng.randint(0, width - patch_w)
    dy = rng.randint(0, height - patch_h)
    patch = donor.crop((sx, sy, sx + patch_w, sy + patch_h))
    image = image.copy()
    image.paste(patch, (dx, dy))
    return image


def _brightness(image: Image.Image, rng: random.Random) -> Image.Image:
    factor = rng.choice([0.55, 0.7, 1.35, 1.6])
    return ImageEnhance.Brightness(image).enhance(factor)


def _jpeg_postprocess(image: Image.Image, rng: random.Random) -> Image.Image:
    quality = rng.choice([10, 20, 30])
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def make_manipulation(
    authentic: np.ndarray,
    donor: np.ndarray,
    tamper_type: str,
    rng: random.Random,
) -> np.ndarray:
    """Apply one controlled manipulation and return an RGB uint8 image."""
    image = Image.fromarray(authentic, mode="RGB")
    donor_image = Image.fromarray(donor, mode="RGB")
    if tamper_type == "copy_move":
        result = _copy_move(image, rng)
    elif tamper_type == "splicing":
        result = _splicing(image, donor_image, rng)
    elif tamper_type == "brightness":
        result = _brightness(image, rng)
    elif tamper_type == "jpeg_recompression":
        result = _jpeg_postprocess(image, rng)
    else:
        raise ValueError(f"Unknown tamper type: {tamper_type}")
    return np.asarray(result, dtype=np.uint8)


def build_benchmark(
    raw_root: Path,
    processed_root: Path,
    source_images: int = 3000,
    seed: int = 42,
    force: bool = False,
) -> pd.DataFrame:
    """Create balanced authentic/manipulated files and return their metadata."""
    processed_root.mkdir(parents=True, exist_ok=True)
    images_root = processed_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    csv_path = processed_root / "benchmark.csv"
    if csv_path.exists() and not force:
        return pd.read_csv(csv_path)

    try:
        extracted = download_and_extract_cifar(raw_root, force=False)
        images, labels = load_cifar_train(extracted)
        source_name = "CIFAR-10"
    except (EOFError, pickle.UnpicklingError, tarfile.ReadError, FileNotFoundError, OSError) as exc:
        print(f"Warning: CIFAR-10 archive is incomplete ({exc}); using the fast local digits pilot source.")
        images, labels = load_fast_fallback()
        source_name = "scikit-learn digits fallback (8x8 grayscale, upscaled to 32x32 RGB)"
    source_images = min(int(source_images), len(images))
    rng_np = np.random.default_rng(seed)
    selected = rng_np.choice(len(images), size=source_images, replace=False)
    split_values = np.array(["train"] * source_images, dtype=object)
    shuffled = selected.copy()
    rng_np.shuffle(shuffled)
    train_count = int(source_images * 0.70)
    val_count = int(source_images * 0.15)
    split_values[np.isin(selected, shuffled[:train_count])] = "train"
    split_values[np.isin(selected, shuffled[train_count : train_count + val_count])] = "validation"
    split_values[np.isin(selected, shuffled[train_count + val_count :])] = "test"

    tamper_types = ["copy_move", "splicing", "brightness", "jpeg_recompression"]
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for row_idx, source_idx in enumerate(selected):
        authentic = images[source_idx]
        donor = images[selected[(row_idx + 1) % len(selected)]]
        source_id = f"cifar10_{int(source_idx):05d}"
        split = str(split_values[row_idx])
        authentic_id = f"{row_idx:05d}_authentic"
        authentic_path = images_root / f"{authentic_id}.png"
        Image.fromarray(authentic).save(authentic_path)
        rows.append(
            {
                "sample_id": authentic_id,
                "source_id": source_id,
                "source_class": int(labels[source_idx]),
                "split": split,
                "label": 0,
                "tamper_type": "authentic",
                "image_path": str(authentic_path),
            }
        )
        tamper_type = tamper_types[row_idx % len(tamper_types)]
        tampered = make_manipulation(authentic, donor, tamper_type, rng)
        tampered_id = f"{row_idx:05d}_{tamper_type}"
        tampered_path = images_root / f"{tampered_id}.png"
        Image.fromarray(tampered).save(tampered_path)
        rows.append(
            {
                "sample_id": tampered_id,
                "source_id": source_id,
                "source_class": int(labels[source_idx]),
                "split": split,
                "label": 1,
                "tamper_type": tamper_type,
                "image_path": str(tampered_path),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    metadata = processed_root / "dataset_metadata.txt"
    metadata.write_text(
        "Requested source: CIFAR-10 official URL: " + CIFAR_URL + "\n"
        f"Actual source used: {source_name}\n"
        f"source_images={source_images}\nseed={seed}\n"
        "split=train/validation/test by source image before manipulation\n",
        encoding="utf-8",
    )
    return frame


def make_visual_samples(frame: pd.DataFrame, output_dir: Path, count: int = 6) -> list[tuple[Path, Path, str]]:
    """Save a small authentic/tampered contact-sheet input set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = frame[(frame["split"] == "test") & (frame["label"] == 1)].head(count)
    samples: list[tuple[Path, Path, str]] = []
    for _, row in selected.iterrows():
        source_id = row["source_id"]
        authentic = frame[(frame["source_id"] == source_id) & (frame["label"] == 0)].iloc[0]
        auth_path = output_dir / f"{row['sample_id']}_authentic.png"
        tamper_path = output_dir / f"{row['sample_id']}_tampered.png"
        Image.open(authentic["image_path"]).save(auth_path)
        Image.open(row["image_path"]).save(tamper_path)
        samples.append((auth_path, tamper_path, str(row["tamper_type"])))
    return samples
