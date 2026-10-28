"""Build a reproducible, source-safe image-manipulation benchmark.

Two deliberate changes from the earlier pilot:

1. **Provenance is derived, never assumed.** The corpus name and per-image source
   id come from the loader that actually produced the pixels, so a fallback can
   never be recorded as CIFAR-10.

2. **Fallback is fatal by default.** A truncated download used to degrade the
   benchmark silently; now it raises unless ``allow_fallback=True`` is passed.

Manipulations model the forgery families that matter in the literature -- copy-move,
splicing and object removal -- using alpha-feathered edits with ground-truth masks.
Brightness and JPEG changes are *post-processing*, not forgery, so they are no
longer classes; they live in the robustness sweep in :mod:`train`.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from sources import (
    Corpus,
    DatasetUnavailable,
    assign_grouped_splits,
    content_fingerprint,
    load_cifar10,
    load_digits_fallback,
)

MANIPULATION_FAMILIES = ("copy_move", "splicing", "removal")

# Bump whenever the edits produced by _MANIPULATORS change for the same pixels
# and seed. A stored benchmark carries this tag, so an older CSV cannot be
# mistaken for one this code would produce.
MANIPULATION_VERSION = "2024-05-feathered-copy-move-splice-removal"

# Columns holding filesystem locations, stored relative to the benchmark dir so
# a committed CSV is not tied to one machine's home directory.
PATH_COLUMNS = ("image_path", "mask_path")

# Post-processing conditions are evaluated as robustness sweeps, not as classes.
ROBUSTNESS_CONDITIONS = ("clean", "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30", "jpeg_q20", "resize_half", "blur", "noise")


def _region(size: tuple[int, int], rng: random.Random, min_frac: float = 0.18, max_frac: float = 0.40) -> tuple[int, int, int, int]:
    """Return a random rectangular region sized as a fraction of the image.

    The 18-40% per-axis range puts forged regions at roughly 3-16% of image area,
    which sits inside the range reported for forged regions in copy-move and
    splicing benchmarks.
    """
    width, height = size
    patch_w = int(width * rng.uniform(min_frac, max_frac))
    patch_h = int(height * rng.uniform(min_frac, max_frac))
    patch_w = max(6, min(patch_w, width - 2))
    patch_h = max(6, min(patch_h, height - 2))
    x = rng.randint(0, width - patch_w)
    y = rng.randint(0, height - patch_h)
    return x, y, x + patch_w, y + patch_h


def _feather_radius(shape: tuple[int, int]) -> int:
    """Feather width scaled to image size.

    A fixed radius destroys the edit on small images: a 2 px blur over a 6 px
    region leaves no boundary at all. Real composites are feathered by roughly
    one to three pixels regardless of resolution, so scale the radius instead of
    pinning it.
    """
    short_side = min(shape)
    return int(max(1, min(3, round(short_side / 128))))


def _feather_mask(shape: tuple[int, int], box: tuple[int, int, int, int], blur: int = 1) -> Image.Image:
    """Binary region mask with soft edges, mimicking a hand-composited edit."""
    x0, y0, x1, y1 = box
    mask = Image.new("L", (shape[1], shape[0]), 0)
    mask.paste(255, (x0, y0, x1, y1))
    return mask.filter(ImageFilter.GaussianBlur(radius=blur)) if blur else mask


def _blend(base: Image.Image, patch: Image.Image, mask: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    x0, y0, _, _ = box
    layer = base.copy()
    layer.paste(patch, (x0, y0))
    return Image.composite(layer, base, mask)


def _copy_move(image: Image.Image, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    """Copy a region and re-paste it at a different location, optionally flipped."""
    width, height = image.size
    source_box = _region(image.size, rng)
    patch = image.crop(source_box)
    if rng.random() < 0.5:
        patch = patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.3:
        patch = patch.rotate(rng.choice([90, 180, 270]), expand=True)
        patch = patch.resize((source_box[2] - source_box[0], source_box[3] - source_box[1]))

    pw, ph = patch.size
    if pw >= width or ph >= height:
        return image.copy(), Image.new("L", image.size, 0)
    x = rng.randint(0, width - pw)
    y = rng.randint(0, height - ph)
    target_box = (x, y, x + pw, y + ph)

    mask = _feather_mask(image.size, target_box, blur=_feather_radius(image.size))
    return _blend(image, patch, mask, target_box), mask


def _match_colour(patch: Image.Image, host: Image.Image, strength: float) -> Image.Image:
    """Partially transfer host colour statistics onto the donor patch (splicing)."""
    patch_array = np.asarray(patch, dtype=np.float32)
    host_array = np.asarray(host, dtype=np.float32)
    for channel in range(3):
        patch_mean = patch_array[..., channel].mean()
        patch_std = patch_array[..., channel].std() + 1e-6
        host_mean = host_array[..., channel].mean()
        host_std = host_array[..., channel].std() + 1e-6
        shifted = (patch_array[..., channel] - patch_mean) * (host_std / patch_std) + host_mean
        patch_array[..., channel] = (1 - strength) * patch_array[..., channel] + strength * shifted
    return Image.fromarray(np.clip(patch_array, 0, 255).astype(np.uint8))


def _splicing(image: Image.Image, donor: Image.Image, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    """Insert content taken from a different source image."""
    source_box = _region(donor.size, rng)
    patch = donor.crop(source_box)
    patch = _match_colour(patch, image, strength=rng.uniform(0.0, 0.6))

    width, height = image.size
    pw, ph = patch.size
    if pw >= width or ph >= height:
        return image.copy(), Image.new("L", image.size, 0)
    x = rng.randint(0, width - pw)
    y = rng.randint(0, height - ph)
    target_box = (x, y, x + pw, y + ph)

    mask = _feather_mask(image.size, target_box, blur=_feather_radius(image.size))
    return _blend(image, patch, mask, target_box), mask


def _removal(image: Image.Image, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    """Simulate naive object removal: fill a region from its surroundings."""
    box = _region(image.size, rng, min_frac=0.2, max_frac=0.4)
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    donor_left = max(0, x0 - max(6, width // 2))
    donor_top = max(0, y0 - max(6, height // 2))
    patch = image.crop((donor_left, donor_top, donor_left + width, donor_top + height))
    if patch.size != (width, height):
        patch = patch.resize((width, height))
    patch = patch.filter(ImageFilter.GaussianBlur(radius=1.5))
    heal = image.copy()
    heal.paste(patch, (x0, y0))
    mask = _feather_mask(image.size, box, blur=_feather_radius(image.size))
    return Image.composite(heal, image, mask), mask


_MANIPULATORS = {
    "copy_move": _copy_move,
    "splicing": _splicing,
    "removal": _removal,
}

# Families that take a donor image as their second argument.
_NEEDS_DONOR = {"splicing"}


def load_corpus(
    raw_root: Path,
    source_images: int,
    seed: int,
    dataset: str = "cifar10",
    allow_fallback: bool = False,
    force_download: bool = False,
) -> Corpus:
    """Load the requested authentic-image corpus, failing loudly by default."""
    if dataset == "cifar10":
        try:
            return load_cifar10(raw_root, source_images, seed=seed, force_download=force_download)
        except DatasetUnavailable:
            if not allow_fallback:
                raise
            print(
                "WARNING: CIFAR-10 unavailable and --allow-fallback was passed. "
                "Using scikit-learn digits. Results are NOT comparable to a natural-image run."
            )
            return load_digits_fallback(source_images, seed=seed)
    if dataset == "digits":
        print("NOTE: using scikit-learn digits. This is a synthetic pilot corpus, not natural images.")
        return load_digits_fallback(source_images, seed=seed)
    raise DatasetUnavailable(f"Unknown authentic corpus {dataset!r}; expected 'cifar10' or 'digits'.")


def store_relative(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Rewrite path columns relative to ``root`` before the CSV is written.

    Absolute paths make a committed benchmark point at one machine's home
    directory: unusable to anyone else and needlessly revealing. A path outside
    ``root`` is left absolute, because a wrong relative path is worse than an
    obvious absolute one.
    """
    frame = frame.copy()
    resolved_root = root.resolve()

    def relativise(value) -> str:
        if not isinstance(value, str) or not value:
            return ""
        path = Path(value)
        try:
            return path.resolve().relative_to(resolved_root).as_posix()
        except ValueError:
            return value

    for column in PATH_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].map(relativise)
    return frame


def _load_absolute(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Resolve stored relative paths against the directory holding the CSV."""
    frame = frame.copy()

    def absolutise(value) -> str:
        if not isinstance(value, str) or not value:
            return ""
        path = Path(value)
        return str(path if path.is_absolute() else (root / path).resolve())

    for column in PATH_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].map(absolutise)
    return frame


def _content_fingerprint(frame: pd.DataFrame) -> str:
    """Digest of the images referenced by a frame, in row order."""
    return content_fingerprint(frame["image_path"].astype(str))


def _cache_matches(
    csv_path: Path,
    processed_root: Path,
    dataset: str,
    families: tuple[str, ...],
    source_images: int,
    seed: int,
) -> bool:
    """True only when the cached benchmark was built for this exact request.

    Without this, switching ``--dataset`` or ``--seed`` silently reuses the
    previous corpus, which is the same class of provenance bug as mislabelled
    source ids. The generator version and a content digest are checked too, so
    a CSV written by older manipulation code, or one whose images were deleted
    or replaced, is rebuilt instead of trusted.
    """
    metadata_path = processed_root / "dataset_metadata.txt"
    if not metadata_path.exists():
        return False
    try:
        cached = pd.read_csv(csv_path)
    except Exception:  # noqa: BLE001 - a corrupt cache should simply be rebuilt
        return False
    if not {"dataset", "mask_path", "image_path", "label", "source_id"}.issubset(cached.columns):
        return False
    if cached.empty:
        return False

    fields = {}
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    expected = {
        "dataset_key": dataset,
        "families": ",".join(families),
        "source_images": str(source_images),
        "seed": str(seed),
        "manipulation_version": MANIPULATION_VERSION,
        "rows": str(len(cached)),
    }
    if any(fields.get(key) != value for key, value in expected.items()):
        return False

    absolute = _load_absolute(cached, processed_root)
    if _content_fingerprint(absolute) != fields.get("content_digest"):
        return False
    return all(Path(path).is_file() for path in absolute["image_path"].astype(str))


def build_benchmark(
    raw_root: Path,
    processed_root: Path,
    source_images: int = 3000,
    seed: int = 42,
    force: bool = False,
    dataset: str = "cifar10",
    allow_fallback: bool = False,
    families: tuple[str, ...] = MANIPULATION_FAMILIES,
    force_download: bool = False,
) -> pd.DataFrame:
    """Create balanced authentic/manipulated files plus masks; return their metadata."""
    processed_root.mkdir(parents=True, exist_ok=True)
    images_root = processed_root / "images"
    masks_root = processed_root / "masks"
    images_root.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)
    csv_path = processed_root / "benchmark.csv"

    unknown = set(families) - set(_MANIPULATORS)
    if unknown:
        raise ValueError(f"Unknown manipulation families: {sorted(unknown)}; expected {list(_MANIPULATORS)}")

    if csv_path.exists() and not force:
        cached = pd.read_csv(csv_path)
        if _cache_matches(csv_path, processed_root, dataset, families, source_images, seed):
            return _load_absolute(cached, processed_root)
        print(
            "Cached benchmark does not match the requested configuration "
            f"(dataset={dataset!r}, seed={seed}); rebuilding."
        )

    corpus = load_corpus(
        raw_root=raw_root,
        source_images=source_images,
        seed=seed,
        dataset=dataset,
        allow_fallback=allow_fallback,
        force_download=force_download,
    )
    count = len(corpus)
    rng = random.Random(seed)

    rows: list[dict[str, object]] = []
    for index in range(count):
        authentic = corpus.images[index]
        source_id = str(corpus.source_ids[index])
        authentic_id = f"{index:05d}_authentic"
        authentic_path = images_root / f"{authentic_id}.png"
        Image.fromarray(authentic).save(authentic_path)
        rows.append(
            {
                "sample_id": authentic_id,
                "source_id": source_id,
                "dataset": corpus.name,
                "source_class": int(corpus.labels[index]),
                "label": 0,
                "tamper_type": "authentic",
                "image_path": str(authentic_path.resolve()),
                "mask_path": "",
            }
        )

        family = families[index % len(families)]
        image = Image.fromarray(authentic, mode="RGB")
        donor_index = (index + 1 + rng.randrange(max(1, count - 1))) % count
        donor = Image.fromarray(corpus.images[donor_index], mode="RGB")
        manipulate = _MANIPULATORS[family]
        if family in _NEEDS_DONOR:
            manipulated, mask = manipulate(image, donor, rng)
        else:
            manipulated, mask = manipulate(image, rng)

        tampered_id = f"{index:05d}_{family}"
        tampered_path = images_root / f"{tampered_id}.png"
        mask_path = masks_root / f"{tampered_id}.png"
        manipulated.save(tampered_path)
        mask.save(mask_path)
        rows.append(
            {
                "sample_id": tampered_id,
                "source_id": source_id,
                "dataset": corpus.name,
                "source_class": int(corpus.labels[index]),
                "label": 1,
                "tamper_type": family,
                "image_path": str(tampered_path.resolve()),
                "mask_path": str(mask_path.resolve()),
            }
        )

    frame = assign_grouped_splits(pd.DataFrame(rows), seed=seed)

    # A source group can only span one fold, but authentic and tampered rows of a
    # group must share that fold; assign_grouped_splits guarantees both.
    store_relative(frame, processed_root).to_csv(csv_path, index=False)
    (processed_root / "dataset_metadata.txt").write_text(
        "corpus=" + corpus.name + "\n"
        f"dataset_key={dataset}\n"
        f"source_images={count}\n"
        f"seed={seed}\n"
        f"rows={len(frame)}\n"
        f"manipulation_version={MANIPULATION_VERSION}\n"
        f"content_digest={_content_fingerprint(frame)}\n"
        f"families={','.join(families)}\n"
        "split=train/validation/test grouped by source_id before manipulation\n"
        "manipulations=synthetic (alpha-feathered), with ground-truth masks\n",
        encoding="utf-8",
    )
    return frame


def make_visual_samples(frame: pd.DataFrame, output_dir: Path, count: int = 6) -> list[tuple[Path, Path, str]]:
    """Save a small authentic/manipulated contact-sheet input set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = frame[(frame["split"] == "test") & (frame["label"] == 1)].head(count)
    samples: list[tuple[Path, Path, str]] = []
    for _, row in selected.iterrows():
        authentic = frame[(frame["source_id"] == row["source_id"]) & (frame["label"] == 0)].iloc[0]
        auth_path = output_dir / f"{row['sample_id']}_authentic.png"
        tamper_path = output_dir / f"{row['sample_id']}_tampered.png"
        Image.open(authentic["image_path"]).save(auth_path)
        Image.open(row["image_path"]).save(tamper_path)
        samples.append((auth_path, tamper_path, str(row["tamper_type"])))
    return samples
