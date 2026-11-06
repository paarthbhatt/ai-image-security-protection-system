"""Build the modified CASIA v2.0 variant used by the base paper.

Why this exists
---------------
Our CASIA v2.0 archive is 7,491 authentic (99.3% JPEG, 54 BMP) plus 5,123
tampered (3,059 TIFF, 2,064 JPEG). File format therefore almost separates the
classes on its own, which is a cue about how a public archive was packaged, not
about forgery. Section 4.1.3 of Yadav & Vishwakarma (2024), the base paper of
this project, removes that cue before experimenting:

    "CASIA v2.0 [...] containing 7491 original and 5123 tampered images.
    However, over half of the tampered images are uncompressed (TIFF or BMP
    format). Since the proposed splice detection framework utilizes compression
    artifacts of image samples, the uncompressed format images were compressed
    with a random quality factor. This modified variant of the CASIA v2.0
    dataset is used for experimentation. [...] 5123 original images are chosen
    from the original CASIA v2.0 dataset for experimentation to maintain class
    balance."

Two details are not stated, and both are resolved here rather than guessed at
silently:

*   **The quality-factor range.** The paper says "a random quality factor" and,
    for its own datasets, argues that a random *integer* QF is richer than the
    step-size-5 grids used by earlier work. It never gives bounds. This builder
    uses uniform integers over ``[QF_LOW, QF_HIGH]`` and writes the value drawn
    for every image into the manifest, so the choice is visible and replaceable.
*   **Whether the 54 BMP originals are converted too.** The sentence is
    triggered by tampered images but names "the uncompressed format images",
    which covers both classes. Converting only the tampered ones would leave
    "is BMP" true for 54 authentic images and zero tampered ones - a perfectly
    pure class cue, a worse shortcut than the one being removed. So this builder
    converts every uncompressed image, authentic included. ``--tampered-only``
    reproduces the narrow reading for anyone who wants the comparison.

Determinism
-----------
The QF for an image is drawn from a hash of the seed and the image stem, not
from a stream position, so adding or dropping images cannot reshuffle the
values assigned to the ones that stay. The authentic subsample is a seeded
permutation over sorted paths for the same reason. Rebuilding with the same
seed reproduces the corpus exactly.

Usage
-----
    python src/modified_casia.py build                 # into data/real/casia2_mod
    python src/modified_casia.py build --qf-low 70 --qf-high 95
    python src/modified_casia.py build --tampered-only
    python src/modified_casia.py verify                # re-open the built manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "real" / "casia2"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "real" / "casia2_mod"
DEFAULT_TARGET_PER_CLASS = 5123

# The paper's own words are "uncompressed (TIFF or BMP format)"; PNG is included
# because it is uncompressed in the same sense and appears in other CASIA mirrors.
UNCOMPRESSED_SUFFIXES = frozenset({".bmp", ".tif", ".tiff", ".png"})
COMPRESSED_SUFFIXES = frozenset({".jpg", ".jpeg"})

QF_LOW = 50
QF_HIGH = 100

MANIFEST_COLUMNS = [
    "image_path",
    "label",
    "tamper_type",
    "source_id",
    "mask_path",
    "qf",
    "converted",
    "original_path",
]


def draw_quality_factor(stem: str, seed: int, low: int = QF_LOW, high: int = QF_HIGH) -> int:
    """A stable pseudo-random integer QF for one image name.

    Hash-based rather than stream-based: the value depends on (seed, stem) only,
    so it does not move when unrelated images enter or leave the build. Modulo
    bias across a 64-bit draw and a range this small is not measurable.
    """
    span = high - low + 1
    if span < 1:
        raise ValueError(f"empty quality-factor range {low}..{high}")
    digest = hashlib.blake2b(f"{seed}:{stem}".encode("utf-8"), digest_size=8).digest()
    return low + int.from_bytes(digest, "big") % span


def is_uncompressed(path: str | Path) -> bool:
    return Path(str(path)).suffix.lower() in UNCOMPRESSED_SUFFIXES


def _relative_to(target: str, base: Path) -> str:
    """Manifest paths are relative to the manifest's own directory.

    ``load_manifest`` resolves them against the manifest parent, so a relative
    path here keeps the corpus portable and free of one machine's home directory
    even when it points at the untouched source archive next door.
    """
    resolved = Path(str(target)).resolve()
    try:
        return os.path.relpath(resolved, base).replace(os.sep, "/")
    except ValueError:
        return str(resolved)


def convert_image(source: Path, destination: Path, quality: int) -> tuple[int, int]:
    """Re-encode one uncompressed image as JPEG at ``quality``.

    Returns the pixel dimensions, which the caller checks against the source:
    masks are indexed by shape, so a resize here would silently misalign every
    localisation label.
    """
    with Image.open(source) as handle:
        width, height = handle.size
        pixels = handle.convert("RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Pillow's default chroma subsampling is left alone on purpose: forcing 4:4:4
    # would hand the detector a cleaner chroma signal than a real camera or a
    # public archive produces, and the cue we are removing is a compression one.
    pixels.save(destination, format="JPEG", quality=int(quality))
    with Image.open(destination) as check:
        if check.size != (width, height):
            raise AssertionError(
                f"conversion changed dimensions for {source.name}: "
                f"{(width, height)} -> {check.size}"
            )
    return width, height


def select_authentic(paths: list[str], target: int, seed: int) -> list[str]:
    """A seeded subsample of the originals, sized to match the tampered class.

    Sorted input, so the selection depends on the candidate set and the seed,
    not on whatever order the directory walk happened to produce.
    """
    ordered = sorted(str(path) for path in paths)
    if target >= len(ordered):
        return ordered
    picks = np.random.default_rng(seed).permutation(len(ordered))[:target]
    return [ordered[int(index)] for index in sorted(picks.tolist())]


def build(
    source_root: Path = DEFAULT_SOURCE,
    output_root: Path = DEFAULT_OUTPUT,
    seed: int = 42,
    target_per_class: int = DEFAULT_TARGET_PER_CLASS,
    qf_low: int = QF_LOW,
    qf_high: int = QF_HIGH,
    tampered_only: bool = False,
    force: bool = False,
) -> dict[str, object]:
    """Materialise the modified corpus and return its audit record."""
    from sources import load_real_forgery_dataset

    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    manifest_path = output_root / "manifest.csv"
    if manifest_path.exists() and not force:
        raise SystemExit(
            f"{manifest_path} already exists. Re-run with --force to rebuild it."
        )

    frame = load_real_forgery_dataset(
        name="casia2", root=source_root, mask_root=source_root, seed=seed
    )
    tampered = frame.loc[frame["label"] == 1]
    authentic = frame.loc[frame["label"] == 0]
    candidates = tampered if tampered_only else frame
    convertible = candidates[candidates["image_path"].map(is_uncompressed)]
    print(
        f"source: {len(frame):,} rows ({len(authentic):,} authentic, "
        f"{len(tampered):,} tampered); {len(convertible):,} uncompressed to convert"
    )

    # Class balance is defined against the tampered count, which the modification
    # leaves untouched: 5,123 tampered images stay 5,123 and the originals are
    # thinned to match, exactly as section 4.1.3 describes.
    target = min(int(target_per_class), len(tampered))
    kept_authentic = select_authentic(list(authentic["image_path"]), target, seed)

    rows: list[dict[str, object]] = []
    converted = 0
    qualities: list[int] = []
    # One dict instead of a boolean mask per image: the authentic loop below
    # needs a record for each kept path, and scanning a 12,614-row frame 5,123
    # times is the kind of cost that looks free until it is not.
    by_path = frame.set_index("image_path")

    def emit(image_path: str, record: pd.Series) -> None:
        nonlocal converted
        stem = Path(str(image_path)).stem
        # The mode decides what is convertible; the suffix only decides how.
        # Re-deriving eligibility from the path here would convert the BMP
        # originals even under --tampered-only, which is the reading that
        # exists precisely to leave them alone.
        eligible = is_uncompressed(image_path) and not (
            tampered_only and int(record["label"]) == 0
        )
        if eligible:
            quality = draw_quality_factor(stem, seed, qf_low, qf_high)
            destination = output_root / "converted" / f"{stem}.jpg"
            convert_image(Path(str(image_path)), destination, quality)
            converted += 1
            qualities.append(quality)
            stored, was_converted, qf_value = str(destination), True, quality
        else:
            stored, was_converted, qf_value = str(image_path), False, ""
        rows.append(
            {
                "image_path": _relative_to(stored, output_root),
                "label": int(record["label"]),
                # tamper_type and source_id come from the loaded frame, never
                # re-derived from the new path: a converted image sits in a
                # different directory, and re-running the scene heuristics on it
                # would break the linkage that keeps a photograph and its splice
                # in the same fold.
                "tamper_type": str(record["tamper_type"]),
                "source_id": str(record["source_id"]),
                "mask_path": _relative_to(record["mask_path"], output_root)
                if str(record["mask_path"])
                else "",
                "qf": qf_value,
                "converted": was_converted,
                "original_path": _relative_to(image_path, output_root),
            }
        )

    for path in kept_authentic:
        emit(path, by_path.loc[path])
    for _, record in tampered.iterrows():
        emit(str(record["image_path"]), record)

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    summary: dict[str, object] = {
        "seed": int(seed),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "qf_range": [int(qf_low), int(qf_high)],
        "convert_tampered_only": bool(tampered_only),
        "rows": int(len(manifest)),
        "authentic": int((manifest["label"] == 0).sum()),
        "tampered": int((manifest["label"] == 1).sum()),
        "converted": int(converted),
        "scenes": int(manifest["source_id"].nunique()),
        "masks_present": int((manifest["mask_path"].astype(str) != "").sum()),
        "tamper_type_counts": manifest["tamper_type"].value_counts().to_dict(),
        "qf_min": int(min(qualities)) if qualities else None,
        "qf_max": int(max(qualities)) if qualities else None,
        "qf_mean": float(np.mean(qualities)) if qualities else None,
        "qf_histogram": {
            str(quality): qualities.count(quality)
            for quality in sorted(set(qualities))
        } if qualities else {},
        "manifest_sha256": _sha256(manifest_path),
    }
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(output_root: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Re-open a built corpus and check what the loader will actually see."""
    from sources import load_real_forgery_dataset

    output_root = Path(output_root).expanduser().resolve()
    frame = load_real_forgery_dataset(name="casia2_mod", root=output_root, seed=42)
    counts = frame["label"].value_counts().to_dict()
    missing = [
        str(path) for path in frame["image_path"] if not Path(str(path)).is_file()
    ]
    suffixes = (
        pd.Series([Path(str(p)).suffix.lower() for p in frame["image_path"]])
        .value_counts()
        .to_dict()
    )
    return {
        "rows": int(len(frame)),
        "labels": {str(k): int(v) for k, v in counts.items()},
        "missing_files": len(missing),
        "suffixes": suffixes,
        "scenes": int(frame["source_id"].nunique()),
        "splits": frame["split"].value_counts().to_dict(),
        "train_test_scene_overlap": _overlap(frame),
    }


def _overlap(frame: pd.DataFrame) -> int:
    train = set(frame.loc[frame["split"] == "train", "source_id"])
    test = set(frame.loc[frame["split"] == "test", "source_id"])
    return len(train & test)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["build", "verify"], nargs="?", default="build")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-per-class", type=int, default=DEFAULT_TARGET_PER_CLASS)
    parser.add_argument("--qf-low", type=int, default=QF_LOW)
    parser.add_argument("--qf-high", type=int, default=QF_HIGH)
    parser.add_argument("--tampered-only", action="store_true",
                        help="Follow the narrow reading of section 4.1.3 and convert only "
                             "uncompressed tampered images, leaving BMP originals alone.")
    parser.add_argument("--force", action="store_true", help="Rebuild over an existing manifest.")
    parser.add_argument("--report", default=str(PROJECT_ROOT / "reports" / "modified_casia_build.json"),
                        help="Where to write the audit record (empty string to skip).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "verify":
        report = verify(Path(args.output_root))
        print(json.dumps(report, indent=2))
        return 0

    summary = build(
        source_root=Path(args.source_root),
        output_root=Path(args.output_root),
        seed=args.seed,
        target_per_class=args.target_per_class,
        qf_low=args.qf_low,
        qf_high=args.qf_high,
        tampered_only=args.tampered_only,
        force=args.force,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "qf_histogram"}, indent=2))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"audit record: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
