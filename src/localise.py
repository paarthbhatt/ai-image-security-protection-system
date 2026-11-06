"""Region-level (localisation) evaluation against stored ground-truth masks.

Image-level scoring answers *was this edited*. Localisation answers *where*, and
is scored with Dice and IoU -- the axis a reviewer in this field expects and the
one the base paper does not report.

Two localisers, deliberately different in kind:

``tiled``
    The trained image-level detector itself, run over overlapping tiles. The
    tile crops are scored by the same model, so no retraining is needed and
    every classical model in the zoo gets a localisation number. Its weakness
    is stated rather than hidden: a model fitted on whole photographs sees
    context a 128-pixel crop lacks, so this measures how far global forensic
    evidence transfers to regions.

``ela``
    Error-level-analysis tile energy, z-normalised within the image and
    thresholded. A classical reference that uses no model at all, so a reviewer
    can see whether the learned detector localises better than the textbook
    trick. If it does not, that is the finding.

Authentic images carry no mask, because there is nothing to locate. They are
scored against an all-zero ground truth, which makes the meaningful number their
false-positive area rather than a Dice coefficient -- and an honest localiser
should be embarrassed by the former, not proud of the latter.

Region thresholds are chosen on validation images only, never on test, and each
split is capped and seeded because tile extraction is the most expensive thing in
this repository.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from features import extract_features

TILE_DEFAULT = 128
STRIDE_DEFAULT = 64
SUBSET_DEFAULT = 240
ELA_QUALITY = 90

# Swept on validation and chosen by mean Dice. Coarse on purpose: the region task
# is noisy, and a fine grid fits the subset instead of the detector.
TILE_THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
ELA_Z_THRESHOLD_GRID = np.round(np.arange(0.25, 3.01, 0.25), 2)


def _tile_bands(length: int, tile: int, stride: int) -> list[int]:
    """Tile start offsets along one axis, always reaching the far edge.

    The final band is clamped to ``length - tile`` so the right and bottom
    borders are covered, at the cost of overlapping the neighbouring tile.
    """
    if length <= tile:
        return [0]
    bands = list(range(0, length - tile + 1, stride))
    if bands[-1] != length - tile:
        bands.append(length - tile)
    return bands


def _ela_residual(gray: np.ndarray) -> np.ndarray:
    """Absolute grey-level change caused by one further JPEG pass at q90."""
    image = Image.fromarray(np.asarray(gray, dtype=np.uint8), mode="L")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=ELA_QUALITY)
    buffer.seek(0)
    with Image.open(buffer) as handle:
        back = np.asarray(handle.convert("L"), dtype=np.float32)
    return np.abs(back - np.asarray(gray, dtype=np.float32))


def _tile_maps(
    image: np.ndarray, tile: int, stride: int
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Feature vector and ELA energy per tile, laid out on the tile grid.

    Both localisers come from one pass, so the expensive feature extraction is
    paid for once per image.
    """
    height, width = image.shape[:2]
    ys = _tile_bands(height, tile, stride)
    xs = _tile_bands(width, tile, stride)
    gray = np.asarray(Image.fromarray(image.astype(np.uint8)).convert("L"), dtype=np.uint8)
    energy = _ela_residual(gray)

    vectors: list[np.ndarray] = []
    ela = np.empty((len(ys), len(xs)), dtype=np.float32)
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            vectors.append(extract_features(image[y : y + tile, x : x + tile]))
            ela[row, col] = float(energy[y : y + tile, x : x + tile].mean())
    return np.asarray(vectors, dtype=np.float32), ela, ys, xs


def _grid_to_image(
    grid: np.ndarray, shape: tuple[int, int], ys: list[int], xs: list[int]
) -> np.ndarray:
    """Extend each tile's value over the pixels it is responsible for.

    A tile owns the area from its origin to the next origin, and the last band
    owns everything up to the edge, which is what the clamped bands above are
    there to make true.
    """
    height, width = shape
    out = np.zeros((height, width), dtype=grid.dtype)
    for row, y in enumerate(ys):
        y_end = ys[row + 1] if row + 1 < len(ys) else height
        for col, x in enumerate(xs):
            x_end = xs[col + 1] if col + 1 < len(xs) else width
            out[y:y_end, x:x_end] = grid[row, col]
    return out


def _mask_to_size(mask_path: object, shape: tuple[int, int]) -> np.ndarray | None:
    """Ground-truth mask as a boolean array at the image's own resolution."""
    if not isinstance(mask_path, str) or not mask_path:
        return None
    path = Path(mask_path)
    if not path.is_file():
        return None
    with Image.open(path) as handle:
        mask = np.asarray(handle.convert("L").resize((shape[1], shape[0]), Image.NEAREST))
    return mask > 127


def _overlap_scores(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Dice, IoU, pixel accuracy and false-positive area between two boolean images.

    Dice and IoU are 0 when both masks are empty: agreeing that nothing was
    tampered with is a correct image-level judgement, not a perfect region
    score, and reporting 1.0 there would inflate the mean on authentic images.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    clean_area = int(np.logical_or(~gt, gt).sum()) - int(gt.sum())
    dice = 0.0 if tp == fp == fn == 0 else (2 * tp) / (2 * tp + fp + fn)
    iou = 0.0 if tp == fp == fn == 0 else tp / (tp + fp + fn)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "pixel_accuracy": float((tp + tn) / pred.size) if pred.size else 0.0,
        "hit": float(tp > 0),
        "predicted_area": float(pred.mean()),
        "true_area": float(gt.mean()),
        "false_positive_area": float(fp / max(1, clean_area)),
    }


def _tile_worker(
    image_path: str, mask_path: object, tile: int, stride: int, assume_clean: bool
) -> dict[str, object] | None:
    """One image's tile features, energy map and ground truth.

    Module level, and given no model, so it can run in a process pool. The
    detector is applied afterwards in one batch instead.
    """
    if not Path(image_path).is_file():
        return None
    with Image.open(image_path) as handle:
        image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    gt = _mask_to_size(mask_path, image.shape[:2])
    if gt is None:
        if not assume_clean:
            return None
        gt = np.zeros(image.shape[:2], dtype=bool)
    vectors, ela, ys, xs = _tile_maps(image, tile, stride)
    return {"vectors": vectors, "ela": ela, "ys": ys, "xs": xs, "gt": gt, "shape": image.shape[:2]}


def _collect(
    model,
    rows: pd.DataFrame,
    tile: int,
    stride: int,
    workers: int | None = None,
) -> list[dict[str, object]]:
    """Tile scores for every readable row, features extracted across processes."""
    tasks: list[tuple[str, object, bool]] = []
    kept: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        assume_clean = int(row["label"]) == 0
        mask_path = row.get("mask_path", "")
        if not assume_clean and (not isinstance(mask_path, str) or mask_path == ""):
            continue
        tasks.append((str(row["image_path"]), mask_path, assume_clean))
        kept.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "tamper_type": str(row.get("tamper_type", "")),
                "label": int(row["label"]),
            }
        )
    if not tasks:
        return []

    count = workers if workers is not None else max(1, min(11, (len(tasks) // 8) or 1))
    if count > 1:
        from joblib import Parallel, delayed

        outputs = Parallel(n_jobs=count, prefer="processes", batch_size=4)(
            delayed(_tile_worker)(path, mask, tile, stride, clean) for path, mask, clean in tasks
        )
    else:
        outputs = [_tile_worker(path, mask, tile, stride, clean) for path, mask, clean in tasks]

    # ``outputs`` is one entry per task, with None where an image was unreadable,
    # so it pairs with ``kept`` -- never with ``rows``, which was filtered.
    records: list[dict[str, object]] = []
    for meta, output in zip(kept, outputs):
        if output is None:
            continue
        probabilities = np.asarray(
            model.predict_proba(output["vectors"])[:, 1], dtype=np.float32
        )
        records.append(
            {
                **meta,
                "shape": output["shape"],
                "ys": output["ys"],
                "xs": output["xs"],
                "tile_prob": probabilities.reshape(len(output["ys"]), len(output["xs"])),
                "ela_energy": output["ela"],
                "gt": output["gt"],
            }
        )
    print(f"  localised {len(records)} images", flush=True)
    return records


def _prepare_ela_z(records: list[dict]) -> None:
    """Z-normalise tile energy within each image so one threshold is comparable."""
    for record in records:
        energy = record["ela_energy"].astype(np.float32)
        spread = float(energy.std())
        record["ela_z"] = (energy - float(energy.mean())) / (spread if spread > 1e-6 else 1.0)


def _mean_dice(records: list[dict], grid_map_key: str, value: float) -> float:
    """Mean Dice over forged images only, at one candidate threshold."""
    scores = [
        _overlap_scores(
            _grid_to_image(record[grid_map_key] >= value, record["shape"], record["ys"], record["xs"]),
            record["gt"],
        )["dice"]
        for record in records
        if record["label"] == 1
    ]
    return float(np.mean(scores)) if scores else 0.0


def _best_threshold(records: list[dict], grid: np.ndarray, key: str) -> tuple[float, float]:
    """The grid value maximising mean validation Dice, and that score."""
    if not any(record["label"] == 1 for record in records):
        return float(grid[len(grid) // 2]), 0.0
    values = [_mean_dice(records, key, float(value)) for value in grid]
    best = int(np.argmax(values))
    return float(grid[best]), values[best]


def _subset(frame: pd.DataFrame, split_mask: np.ndarray, limit: int, seed: int) -> pd.DataFrame:
    """Deterministic slice of one split, stratified by label and tamper family."""
    rows = frame.loc[split_mask].copy()
    if not limit or len(rows) <= limit:
        return rows.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    picked: list = []
    keys = rows["label"].astype(str) + "|" + rows["tamper_type"].astype(str)
    for _, group in rows.groupby(keys, sort=True):
        take = max(1, int(round(limit * len(group) / len(rows))))
        take = min(take, len(group))
        positions = np.sort(rng.choice(len(group), size=take, replace=False))
        picked.extend(group.index[positions].tolist())
    rows = rows.loc[picked]
    if len(rows) > limit:
        rows = rows.iloc[np.sort(rng.choice(len(rows), size=limit, replace=False))]
    return rows.reset_index(drop=True)


def _summarise(
    records: list[dict], key: str, threshold: float, method: str
) -> dict[str, object]:
    """Aggregate region scores, split by tamper family, with an authentic check."""
    if not records:
        return {"method": method, "images": 0}
    per_image = []
    for record in records:
        pred = _grid_to_image(record[key] >= threshold, record["shape"], record["ys"], record["xs"])
        scores = _overlap_scores(pred, record["gt"])
        scores.update(
            sample_id=record["sample_id"],
            tamper_type=record["tamper_type"],
            label=record["label"],
        )
        per_image.append(scores)
    table = pd.DataFrame(per_image)
    forged = table[table["label"] == 1]
    authentic = table[table["label"] == 0]
    by_family = (
        {
            family: {
                "images": int(len(group)),
                "dice": float(group["dice"].mean()),
                "iou": float(group["iou"].mean()),
                "hit_rate": float(group["hit"].mean()),
            }
            for family, group in forged.groupby("tamper_type")
        }
        if len(forged)
        else {}
    )
    return {
        "method": method,
        "images": int(len(table)),
        "forged_images": int(len(forged)),
        "authentic_images": int(len(authentic)),
        "dice": float(forged["dice"].mean()) if len(forged) else 0.0,
        "iou": float(forged["iou"].mean()) if len(forged) else 0.0,
        "hit_rate": float(forged["hit"].mean()) if len(forged) else 0.0,
        "pixel_accuracy": float(forged["pixel_accuracy"].mean()) if len(forged) else 0.0,
        "authentic_false_positive_area": (
            float(authentic["false_positive_area"].mean()) if len(authentic) else 0.0
        ),
        "region_threshold": float(threshold),
        "by_family": by_family,
    }


def evaluate_localisation(
    model,
    frame: pd.DataFrame,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    seed: int = 42,
    tile: int = TILE_DEFAULT,
    stride: int = STRIDE_DEFAULT,
    limit: int = SUBSET_DEFAULT,
    workers: int | None = None,
    region_thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    """Score tampered-region localisation, with thresholds set on validation only.

    ``region_thresholds`` overrides the validation sweep with thresholds decided
    elsewhere -- required for cross-dataset testing, where tuning on the external
    corpus would leak its distribution into our operating point.
    """
    if "mask_path" not in frame.columns:
        return {"available": False, "reason": "no mask_path column"}
    if not (frame["mask_path"].astype(str) != "").any():
        return {"available": False, "reason": "this corpus stores no ground-truth masks"}

    test_rows = _subset(frame, test_mask, limit, seed)
    if region_thresholds is not None:
        tile_threshold = float(region_thresholds["tiled"])
        ela_threshold = float(region_thresholds["ela"])
        # None rather than NaN: this goes straight into results.json, where a
        # non-finite float would serialise as invalid JSON.
        val_dice = None
        val_ela_dice = None
        threshold_source = "inherited from the training corpus"
        print(
            f"Localisation: {len(test_rows)} external images (tile {tile}, stride {stride}), "
            f"thresholds inherited ({tile_threshold:.2f} tiled / {ela_threshold:.2f} ELA)",
            flush=True,
        )
    else:
        val_rows = _subset(frame, val_mask, limit, seed)
        print(
            f"Localisation: {len(val_rows)} validation + {len(test_rows)} test images "
            f"(tile {tile}, stride {stride})",
            flush=True,
        )
        val_records = _collect(model, val_rows, tile, stride, workers)
        if not val_records:
            return {"available": False, "reason": "no readable mask/image pairs in validation"}
        _prepare_ela_z(val_records)
        tile_threshold, val_dice = _best_threshold(val_records, TILE_THRESHOLD_GRID, "tile_prob")
        ela_threshold, val_ela_dice = _best_threshold(
            val_records, ELA_Z_THRESHOLD_GRID, "ela_z"
        )
        threshold_source = "chosen on this corpus's validation split"

    test_records = _collect(model, test_rows, tile, stride, workers)
    if not test_records:
        return {"available": False, "reason": "no readable mask/image pairs in the scored split"}
    _prepare_ela_z(test_records)
    return {
        "available": True,
        "tile": tile,
        "stride": stride,
        "subset_limit": limit,
        "seed": seed,
        "validation_dice_tiled": val_dice,
        "validation_dice_ela": val_ela_dice,
        "threshold_source": threshold_source,
        "tiled": _summarise(test_records, "tile_prob", tile_threshold, "tiled"),
        "ela": _summarise(test_records, "ela_z", ela_threshold, "ela"),
        "note": (
            "Region thresholds "
            + threshold_source
            + ". Tiled re-applies the image-level detector to crops rather than "
            "retraining it. Authentic images have no mask and are scored against an "
            "empty region."
        ),
    }
