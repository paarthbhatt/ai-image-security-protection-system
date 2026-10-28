"""Dataset sources: natural-image corpora and real forensic benchmarks.

Two very different things live here and must never be confused:

*Natural corpora* (CIFAR-10, MNIST-style digits) supply **authentic** images only.
Manipulations are synthesised on top of them by :mod:`dataset`.

*Real forensic benchmarks* (CASIA v2.0, CoMoFoD, COVERAGE, DSO-1, ...) supply
**authentic and genuinely manipulated** images that were forged by humans, usually
with ground-truth masks. They are the only valid evidence of real-world
performance; synthetic manipulations are a controlled pilot.

Every loader returns rows with the same schema so downstream code is uniform::

    sample_id   unique string id
    source_id   grouping key; variants of one scene share it (leakage control)
    dataset     which corpus the row came from
    label       0 = authentic, 1 = manipulated
    tamper_type "authentic" | "copy_move" | "splicing" | ... | "unknown"
    image_path  absolute path to the image file
    mask_path   absolute path to a ground-truth mask, or "" when absent
"""

from __future__ import annotations

import hashlib
import os
import pickle
import random
import re
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MEMBERS = [
    "cifar-10-batches-py/data_batch_1",
    "cifar-10-batches-py/data_batch_2",
    "cifar-10-batches-py/data_batch_3",
    "cifar-10-batches-py/data_batch_4",
    "cifar-10-batches-py/data_batch_5",
    "cifar-10-batches-py/test_batch",
    "cifar-10-batches-py/batches.meta",
]

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Folder names used by public forgery benchmarks to mark the authentic half.
AUTHENTIC_HINTS = {"au", "auth", "authentic", "original", "originals", "genuine", "pristine", "real"}
# Folder names used to mark the manipulated half.
TAMPERED_HINTS = {"tp", "tampered", "tamper", "forged", "forgery", "fake", "manipulated", "spliced", "copy-move", "copymove"}
# Folder names that mark ground-truth masks.
MASK_HINTS = {"gt", "mask", "masks", "groundtruth", "ground_truth", "ground-truth"}


class DatasetUnavailable(RuntimeError):
    """Raised when a requested dataset is missing or incomplete.

    Deliberately fatal by default: a silently substituted dataset invalidates every
    number produced afterwards.
    """


@dataclass
class Corpus:
    """An array-backed authentic-image corpus."""

    images: np.ndarray  # N,H,W,C uint8
    labels: np.ndarray  # N int64 (class labels, not authenticity)
    name: str
    source_ids: np.ndarray  # N str, stable per image

    def __len__(self) -> int:
        return int(len(self.images))


# ---------------------------------------------------------------------------
# Natural image corpora (authentic-only)
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cifar_archive_is_complete(archive: Path) -> bool:
    """Return True only if the archive is a readable tar holding all CIFAR members."""
    if not archive.exists() or archive.stat().st_size < 150_000_000:
        return False
    try:
        with tarfile.open(archive, "r:gz") as tar:
            present = set(tar.getnames())
    except (tarfile.TarError, OSError):
        return False
    return all(member in present for member in CIFAR_MEMBERS)


def download_cifar10(raw_root: Path, force: bool = False, verbose: bool = True) -> Path:
    """Download and verify CIFAR-10, returning the directory holding the batches.

    The archive is downloaded to ``.part`` and only renamed once the tar is proven
    readable and complete, so an interrupted transfer can never masquerade as a
    good dataset.
    """
    raw_root.mkdir(parents=True, exist_ok=True)
    archive = raw_root / "cifar-10-python.tar.gz"
    extracted = raw_root / "cifar-10-batches-py"

    if cifar_archive_is_complete(archive) and not force:
        if not extracted.exists() or len(list(extracted.iterdir())) < len(CIFAR_MEMBERS):
            _extract_cifar(archive, raw_root)
        return extracted

    partial = raw_root / "cifar-10-python.tar.gz.part"
    if partial.exists() and not force:
        if verbose:
            print(f"Resuming partial CIFAR-10 download from {partial.stat().st_size / 1e6:.1f} MB ...")
        _download(archive_url=CIFAR_URL, destination=partial, resume=True, verbose=verbose)
    else:
        if verbose:
            print(f"Downloading CIFAR-10 from {CIFAR_URL} ...")
        _download(archive_url=CIFAR_URL, destination=partial, resume=force, verbose=verbose)

    if not cifar_archive_is_complete(partial):
        raise DatasetUnavailable(
            f"CIFAR-10 archive at {partial} is incomplete or corrupt "
            f"({partial.stat().st_size if partial.exists() else 0} bytes). "
            f"Expected ~170 MB. Re-run, or download {CIFAR_URL} manually and place it at {archive}."
        )
    partial.replace(archive)
    _extract_cifar(archive, raw_root)
    return extracted


def _download(archive_url: str, destination: Path, resume: bool, verbose: bool) -> None:
    """Stream a URL to disk with optional byte-range resume and progress output."""
    request = urllib.request.Request(archive_url, headers={"User-Agent": "researchproject/1.0"})
    offset = destination.stat().st_size if (resume and destination.exists()) else 0
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            mode = "ab" if offset and response.status == 206 else "wb"
            if mode == "wb":
                offset = 0
            total = response.headers.get("Content-Length")
            total = (int(total) + offset) if total else 0
            written = offset
            with destination.open(mode) as handle:
                while True:
                    chunk = response.read(1 << 18)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    if verbose and total and written % (8 << 18) < (1 << 18):
                        print(f"  {written / 1e6:7.1f} / {total / 1e6:.1f} MB", flush=True)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and destination.exists():
            return  # range already satisfied: the partial file is complete
        raise


def _extract_cifar(archive: Path, raw_root: Path) -> None:
    staging = raw_root / "_cifar_extract"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(staging)
    origin = staging / "cifar-10-batches-py"
    target = raw_root / "cifar-10-batches-py"
    if target.exists():
        shutil.rmtree(target)
    origin.rename(target)
    shutil.rmtree(staging, ignore_errors=True)


def load_cifar10(raw_root: Path, count: int, seed: int = 42, force_download: bool = False) -> Corpus:
    """Load authentic natural images from the official CIFAR-10 train split."""
    extracted = download_cifar10(raw_root, force=force_download)
    images: list[np.ndarray] = []
    labels: list[int] = []
    for batch_idx in range(1, 6):
        with (extracted / f"data_batch_{batch_idx}").open("rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        data = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
        images.append(data.transpose(0, 2, 3, 1))
        labels.extend(int(value) for value in batch[b"labels"])
    all_images = np.concatenate(images, axis=0)
    all_labels = np.asarray(labels, dtype=np.int64)

    rng = np.random.default_rng(seed)
    count = min(int(count), len(all_images))
    chosen = np.sort(rng.choice(len(all_images), size=count, replace=False))
    return Corpus(
        images=all_images[chosen],
        labels=all_labels[chosen],
        name="CIFAR-10 (natural images, 32x32)",
        source_ids=np.asarray([f"cifar10_{int(i):05d}" for i in chosen], dtype=object),
    )


def load_digits_fallback(count: int, seed: int = 42) -> Corpus:
    """Load scikit-learn digits as an explicitly-labelled emergency pilot corpus."""
    from sklearn.datasets import load_digits
    from PIL import Image

    digits = load_digits()
    gray = digits.images.astype(np.float32)
    gray = (gray - gray.min()) / (gray.max() - gray.min()) * 255.0
    gray = gray.astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    upscaled = np.stack(
        [np.asarray(Image.fromarray(img).resize((32, 32), Image.Resampling.NEAREST)) for img in rgb],
        axis=0,
    )
    rng = np.random.default_rng(seed)
    count = min(int(count), len(upscaled))
    chosen = np.sort(rng.choice(len(upscaled), size=count, replace=False))
    return Corpus(
        images=upscaled[chosen],
        labels=digits.target.astype(np.int64)[chosen],
        name="scikit-learn digits fallback (8x8 grayscale, upscaled to 32x32 RGB)",
        source_ids=np.asarray([f"digits_{int(i):05d}" for i in chosen], dtype=object),
    )


# ---------------------------------------------------------------------------
# Real forensic benchmarks (authentic + human-made manipulations)
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase and strip separators so folder names compare predictably."""
    return re.sub(r"[^a-z]", "", text.lower())


def _hint_match(name: str, hints: set[str]) -> bool:
    """Match a folder name against a hint set without false positives.

    Short hints such as ``au`` or ``tp`` are matched exactly, because as
    substrings they would fire on unrelated words (``au`` occurs inside
    ``default``). Longer hints match as substrings so real-world names like
    ``CASIA 2 Groundtruth`` still resolve.
    """
    normalised = _normalise(name)
    if not normalised:
        return False
    for hint in hints:
        clean = _normalise(hint)
        if not clean:
            continue
        if len(clean) >= 4:
            if clean in normalised:
                return True
        elif normalised == clean:
            return True
    return False


def _iter_images(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not _is_mask_path(path)
    )


def _is_comofod_mask(path: Path) -> bool:
    """True for a CoMoFoD ``<scene>_M`` mask that accompanies a ``<scene>_F`` forgery.

    Without this check the mask files are counted as authentic images, which
    silently corrupts the label balance.
    """
    stem = path.stem
    if not stem.upper().endswith("_M"):
        return False
    base = stem[:-2]
    return any(path.with_name(f"{base}_F{suffix}").is_file() for suffix in IMAGE_SUFFIXES)


def _is_mask_path(path: Path) -> bool:
    if any(_hint_match(part, MASK_HINTS) for part in path.parts[:-1]):
        return True
    if _is_comofod_mask(path):
        return True
    stem = _normalise(path.stem)
    return stem.endswith(("gt", "mask")) or "groundtruth" in stem


def _classify_folder(path: Path) -> str | None:
    """Map a directory path onto 'authentic' / 'tampered' / 'mask', else None."""
    for part in reversed(path.parts):
        if not part:
            continue
        if _hint_match(part, MASK_HINTS):
            return "mask"
        if _hint_match(part, AUTHENTIC_HINTS):
            return "authentic"
        if _hint_match(part, TAMPERED_HINTS):
            return "tampered"
    return None


def _tamper_type_from_name(path: Path) -> str:
    text = " ".join(path.parts).lower()
    if "copy" in text and "move" in text:
        return "copy_move"
    if "splic" in text or "splice" in text:
        return "splicing"
    if "remov" in text or "inpaint" in text:
        return "removal"
    return "unknown_real"


def find_mask_for(image: Path, dataset_root: Path) -> Path | None:
    """Best-effort lookup of a ground-truth mask for a forged image."""
    candidates: list[Path] = []
    for mask_dir in dataset_root.rglob("*"):
        if mask_dir.is_dir() and _hint_match(mask_dir.name, MASK_HINTS):
            for suffix in IMAGE_SUFFIXES:
                candidates.append(mask_dir / f"{image.stem}{suffix}")
                candidates.append(mask_dir / f"{image.stem}_gt{suffix}")
                candidates.append(mask_dir / f"{image.stem}_mask{suffix}")
    # CoMoFoD keeps the mask beside the forged image as <scene>_M.png
    if image.stem.upper().endswith("_F"):
        base = image.stem[:-2]
        for suffix in IMAGE_SUFFIXES:
            candidates.append(image.with_name(f"{base}_M{suffix}"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def discover_layout(root: Path) -> str:
    """Describe how a dataset directory is organised, for error messages."""
    if not root.exists():
        return "missing"
    top = sorted({p.name for p in root.iterdir()})
    return f"{root} -> {top[:12]}"


REQUIRED_MANIFEST_COLUMNS = {"image_path", "label"}


def load_manifest(manifest: Path, name: str, seed: int) -> pd.DataFrame:
    """Load an explicit CSV manifest, bypassing layout auto-detection.

    Any benchmark whose naming convention is not recognised can be described
    once and loaded forever. Relative paths resolve against the manifest's own
    directory. Required columns are ``image_path`` and ``label``; ``tamper_type``,
    ``source_id`` and ``mask_path`` are optional.
    """
    import pandas as pd

    frame = pd.read_csv(manifest)
    missing = REQUIRED_MANIFEST_COLUMNS - set(frame.columns)
    if missing:
        raise DatasetUnavailable(
            f"{name}: manifest {manifest} is missing required columns {sorted(missing)}."
        )

    base = manifest.parent

    def resolve(value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
            return ""
        path = Path(str(value))
        return str((path if path.is_absolute() else base / path).resolve())

    frame = frame.copy()
    frame["image_path"] = frame["image_path"].map(resolve)
    frame["mask_path"] = frame["mask_path"].map(resolve) if "mask_path" in frame else ""
    frame["label"] = frame["label"].astype(int)
    frame["tamper_type"] = (
        frame["tamper_type"].fillna("unknown_real") if "tamper_type" in frame else "unknown_real"
    )
    frame.loc[frame["label"] == 0, "tamper_type"] = "authentic"
    frame["dataset"] = name
    frame["source_id"] = (
        frame["source_id"].astype(str) if "source_id" in frame else frame["image_path"].map(lambda p: Path(p).stem)
    )
    frame["sample_id"] = frame["image_path"].map(lambda p: f"{name}:{Path(p).stem}")

    present = frame[frame["image_path"].map(lambda p: Path(p).is_file())]
    if present.empty:
        raise DatasetUnavailable(f"{name}: no image listed in {manifest} exists on disk.")
    if len(present) < len(frame):
        print(f"  warning: {len(frame) - len(present)} manifest rows point at missing files")

    if not {0, 1}.issubset(set(present["label"])):
        raise DatasetUnavailable(f"{name}: manifest must contain both label 0 and label 1.")

    columns = ["sample_id", "source_id", "dataset", "label", "tamper_type", "image_path", "mask_path"]
    return assign_grouped_splits(present[columns].reset_index(drop=True), seed=seed)


def load_real_forgery_dataset(
    name: str,
    root: Path,
    mask_root: Path | None = None,
    max_images: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load any authentic/tampered benchmark arranged in the usual public layout.

    Works out of the box for CASIA v2.0 (``Au/`` + ``Tp/``), CoMoFoD
    (``*_F.png`` / ``*_M.png`` per scene) and COVERAGE (``image/`` + ``mask/``),
    and for any other dataset that separates authentic from manipulated folders.

    Rows for one scene share a ``source_id`` so grouped splits keep every variant
    of a scene inside a single fold.
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise DatasetUnavailable(
            f"{name}: dataset directory not found at {root}. "
            "Download it manually and unpack it there (see docs/REAL-DATASETS.md)."
        )

    for candidate in (root / "manifest.csv", root / f"{name}_manifest.csv"):
        if candidate.is_file():
            print(f"  using explicit manifest {candidate.name}")
            return load_manifest(candidate, name, seed)

    images = _iter_images(root)
    if not images:
        raise DatasetUnavailable(f"{name}: no images found under {root} ({discover_layout(root)})")

    rows: list[dict[str, object]] = []
    for path in images:
        role = _classify_folder(path.relative_to(root).parent)
        if role == "authentic":
            label, tamper_type = 0, "authentic"
        elif role == "tampered":
            label, tamper_type = 1, _tamper_type_from_name(path)
        elif role == "mask":
            continue
        else:
            # CoMoFoD convention: <scene>_F (forged) / <scene>_M (mask) / <scene> (original)
            stem = path.stem
            if stem.upper().endswith("_F"):
                label, tamper_type = 1, "unknown_real"
            else:
                label, tamper_type = 0, "authentic"

        mask_path = ""
        if label == 1 and mask_root is not None:
            found = find_mask_for(path, Path(mask_root).expanduser().resolve())
            mask_path = str(found) if found else ""

        rows.append(
            {
                "sample_id": f"{name}:{path.relative_to(root).with_suffix('').as_posix()}",
                "source_id": f"{name}:{_scene_key(path, root)}",
                "dataset": name,
                "label": label,
                "tamper_type": tamper_type,
                "image_path": str(path.resolve()),
                "mask_path": mask_path,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise DatasetUnavailable(f"{name}: layout at {root} produced no labelled images.")

    if not (frame["label"] == 0).any() or not (frame["label"] == 1).any():
        raise DatasetUnavailable(
            f"{name}: only one class was found under {root} "
            f"({frame['label'].value_counts().to_dict()}). Authentic and manipulated "
            "images must both be present; check the folder names against "
            "AUTHENTIC_HINTS/TAMPERED_HINTS in src/sources.py."
        )

    frame = assign_grouped_splits(frame, seed=seed)

    if max_images is not None and len(frame) > max_images:
        frame = _stratified_cap(frame, max_images, seed)

    return frame.reset_index(drop=True)


def _scene_key(path: Path, root: Path) -> str:
    """Group key: variants of one scene collapse to a single id."""
    stem = path.stem
    stem = re.sub(r"(_F|_M|_gt|_mask|-gt|_gt\d*)$", "", stem, flags=re.IGNORECASE)
    # CASIA names files Tp_D_CND_... and Au_ani_...; keep the full stem, it is unique.
    return f"{path.parent.relative_to(root).as_posix()}/{stem}"


def _stratified_cap(frame: pd.DataFrame, max_images: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for _, group in frame.groupby(["dataset", "label"], sort=False):
        take = max(1, int(round(max_images * len(group) / len(frame))))
        indices = rng.choice(group.index.to_numpy(), size=min(take, len(group)), replace=False)
        keep.extend(int(i) for i in indices)
    return frame.loc[sorted(keep)]


def content_fingerprint(paths) -> str:
    """Digest of the pixels behind a list of files, cheap enough to run often.

    Sample ids are positional (``00000_authentic``), so regenerating a benchmark
    with a different seed reuses every id while changing every image. Hashing
    size plus the first block of each file catches that; modification times would
    also be invalidated by copying a tree, and hashing whole files would read
    gigabytes on a real-resolution benchmark.
    """
    digest = hashlib.blake2b(digest_size=16)
    for path in paths:
        try:
            stat = os.stat(str(path))
            with open(str(path), "rb") as handle:
                head = handle.read(8192)
        except OSError:
            digest.update(b"missing")
            continue
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(head)
    return digest.hexdigest()


def assign_grouped_splits(
    frame: pd.DataFrame,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> pd.DataFrame:
    """Split by ``source_id`` so no scene leaks across folds."""
    frame = frame.copy()
    groups = np.asarray(sorted(frame["source_id"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    assignment = {}
    for group in groups[:n_train]:
        assignment[group] = "train"
    for group in groups[n_train : n_train + n_val]:
        assignment[group] = "validation"
    for group in groups[n_train + n_val :]:
        assignment[group] = "test"
    frame["split"] = frame["source_id"].map(assignment)
    return frame


def available_real_datasets() -> dict[str, str]:
    """Dataset name -> expected local directory, for CLI discovery."""
    return {
        "casia2": "data/real/casia2",
        "comofod": "data/real/comofod",
        "coverage": "data/real/coverage",
    }
