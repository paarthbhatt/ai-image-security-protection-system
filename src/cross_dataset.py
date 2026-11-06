"""Zero-shot cross-dataset evaluation.

Training and testing on modified CASIA v2.0 answers "can this pipeline learn
this corpus". It cannot answer the question a reviewer in image forensics
actually asks: does the learned evidence transfer to forgeries made by other
hands, on other cameras, under other compression?

So the external corpus is scored exactly once, with nothing about it allowed to
influence the detector:

* the model is the one already fitted on the training corpus;
* the decision threshold is the one chosen on the *training* corpus's
  validation split -- retuning it on the external images would leak them;
* region thresholds are inherited the same way, so localisation is measured at
  the operating point the paper reports;
* every external image lands in the test split, because none of it was trained
  on and there is nothing to validate on.

Any drop against the in-domain number is reported, not smoothed over. A transfer
result that only keeps the good half is not a transfer result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURE_VERSION
from features import extract_frame
from features import feature_names as feature_frame_names
from localise import evaluate_localisation
from sources import content_fingerprint
from sources import load_real_forgery_dataset
from train import evaluate_probabilities


def _cached_matrix(cache_path: Path, frame: pd.DataFrame, names: list[str]) -> np.ndarray | None:
    """Reuse external features only if they describe these exact pixels."""
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            stored_ids = [str(value) for value in data["sample_id"]]
            stored_names = [str(value) for value in data["feature_names"]]
            stored_version = str(data["feature_version"]) if "feature_version" in data else ""
            stored_content = str(data["content_key"]) if "content_key" in data else ""
            matrix = data["X"]
    except Exception:  # noqa: BLE001 - an unreadable cache is a cache miss
        return None
    if stored_version != FEATURE_VERSION or stored_names != names:
        return None
    if stored_ids != list(frame["sample_id"].astype(str)):
        return None
    if stored_content != content_fingerprint(frame["image_path"].astype(str)):
        return None
    return matrix


def cap_variants_per_scene(
    frame: pd.DataFrame, per_scene: int, seed: int = 42
) -> pd.DataFrame:
    """Keep every scene, but bound how many of its near-duplicates are scored.

    CoMoFoD ships roughly forty post-processing variants of each photograph,
    JPEG and blur and noise and colour-reduction on both the original and the
    forgery. Scoring all of them would call 260 photographs a 10,400-sample
    benchmark and shrink every confidence interval by a factor nobody believes.
    Capping per scene, uniformly and seeded, keeps the breadth while reporting an
    honest effective sample size.
    """
    if not per_scene or per_scene <= 0:
        return frame
    rng = np.random.default_rng(seed)
    kept: list = []
    for _, group in frame.groupby("source_id", sort=True):
        for label in (0, 1):
            rows = group[group["label"] == label]
            if rows.empty:
                continue
            take = min(per_scene, len(rows))
            positions = np.sort(rng.choice(len(rows), size=take, replace=False))
            kept.extend(rows.index[positions].tolist())
    return frame.loc[kept].reset_index(drop=True)


def load_external_frame(
    name: str,
    root: Path,
    seed: int = 42,
    max_images: int | None = None,
    variants_per_scene: int = 4,
) -> pd.DataFrame:
    """Load one external corpus and mark every row as unseen test data."""
    root = Path(root).expanduser().resolve()
    frame = load_real_forgery_dataset(
        name=name, root=root, mask_root=root, seed=seed, max_images=max_images
    )
    frame = frame.copy()
    frame["split"] = "test"
    frame = frame.reset_index(drop=True)
    return cap_variants_per_scene(frame, variants_per_scene, seed)


def by_tamper_type(
    frame: pd.DataFrame, y: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, dict[str, object]]:
    """One score per forgery family, always measured against authentic images.

    A family's ROC-AUC compares that family's forgeries with every authentic row
    of the same corpus. Mean probability and recall alone would hide the case
    that matters here: CoMoFoD's readme documents five tamper geometries, a
    translation-offset cue can only address one of them, and a single pooled
    number averages that difference into invisibility.

    Boolean masks are used rather than the grouped frame's indexes, so nothing
    depends on row positions happening to line up with probability positions.
    """
    tamper_types = frame["tamper_type"].astype(str).to_numpy()
    authentic = y == 0
    forged = y == 1
    report: dict[str, dict[str, object]] = {}
    for family in sorted(set(tamper_types[forged])):
        mask = forged & (tamper_types == family)
        subset = mask | authentic
        metrics = evaluate_probabilities(y[subset], probabilities[subset], threshold)
        report[family] = {
            "images": int(subset.sum()),
            "forged": int(mask.sum()),
            "authentic": int(authentic.sum()),
            "roc_auc": metrics["roc_auc"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "mean_probability": float(probabilities[mask].mean()),
            "recall_at_training_threshold": float(metrics["recall"]),
        }
    return report


def evaluate_cross_dataset(
    model,
    threshold: float,
    training_localisation: dict | None,
    datasets: dict[str, Path],
    processed_root: Path,
    seed: int = 42,
    localise_limit: int = 120,
    max_images: int | None = None,
    variants_per_scene: int = 4,
) -> dict[str, object]:
    """Score one fitted detector on each untouched corpus, in-domain style.

    ``training_localisation`` is the in-domain localisation block, whose region
    thresholds are reused here. ``datasets`` maps a dataset key to its local
    directory; a missing directory records the reason instead of aborting the
    run, because the training-corpus evidence is already banked.
    """
    processed_root.mkdir(parents=True, exist_ok=True)
    names = feature_frame_names()
    region_thresholds = None
    if training_localisation and training_localisation.get("available"):
        region_thresholds = {
            "tiled": float(training_localisation["tiled"]["region_threshold"]),
            "ela": float(training_localisation["ela"]["region_threshold"]),
        }

    report: dict[str, object] = {
        "protocol": (
            "zero-shot: model, decision threshold and region thresholds all fixed on "
            "the training corpus; the external corpus contributes no tuning signal"
        ),
        "variants_per_scene": int(variants_per_scene),
        "training_threshold": float(threshold),
        "region_thresholds_inherited": region_thresholds,
        "datasets": {},
    }

    for name, root in datasets.items():
        print(f"\n=== Cross-dataset: {name} ({root}) ===", flush=True)
        try:
            frame = load_external_frame(
                name, root, seed=seed, max_images=max_images,
                variants_per_scene=variants_per_scene,
            )
        except Exception as exc:  # noqa: BLE001 - a missing corpus is a finding, not a crash
            print(f"  unavailable: {exc}", flush=True)
            report["datasets"][name] = {"available": False, "reason": str(exc)}
            continue

        cache_path = processed_root / f"features_cross_{name}.npz"
        matrix = _cached_matrix(cache_path, frame, names)
        if matrix is None:
            print(f"  extracting features for {len(frame)} images ...", flush=True)
            matrix, names = extract_frame(frame)
            np.savez_compressed(
                cache_path,
                X=matrix,
                y=frame["label"].to_numpy(dtype=np.int64),
                split=frame["split"].to_numpy(dtype=str),
                tamper_type=frame["tamper_type"].to_numpy(dtype=str),
                sample_id=frame["sample_id"].to_numpy(dtype=str),
                source_id=frame["source_id"].to_numpy(dtype=str),
                feature_names=np.asarray(names, dtype=str),
                feature_version=np.asarray(FEATURE_VERSION),
                content_key=np.asarray(content_fingerprint(frame["image_path"].astype(str))),
            )
        else:
            print(f"  reusing {cache_path.name}", flush=True)

        y = frame["label"].to_numpy(dtype=np.int64)
        if y.min() == y.max():
            report["datasets"][name] = {
                "available": False,
                "reason": f"external corpus holds a single class (label={int(y[0])})",
            }
            continue

        probabilities = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
        metrics = evaluate_probabilities(y, probabilities, threshold)
        test_mask = np.ones(len(frame), dtype=bool)
        print(
            f"  {name}: ACC {metrics['accuracy']:.3f} "
            f"bal-ACC {metrics['balanced_accuracy']:.3f} "
            f"AUC {metrics['roc_auc']:.3f} MCC {metrics['mcc']:.3f}",
            flush=True,
        )

        by_family = by_tamper_type(frame, y, probabilities, threshold)
        for family, value in by_family.items():
            print(
                f"    {family:<26} forged {value['forged']:<4} AUC "
                f"{value['roc_auc']:.3f} recall "
                f"{value['recall_at_training_threshold']:.3f}",
                flush=True,
            )

        localisation = (
            evaluate_localisation(
                model,
                frame,
                val_mask=np.zeros(len(frame), dtype=bool),
                test_mask=test_mask,
                seed=seed,
                limit=int(localise_limit),
                region_thresholds=region_thresholds,
            )
            if localise_limit and region_thresholds is not None
            else {
                "available": False,
                "reason": (
                    "localisation skipped by request"
                    if not localise_limit
                    else "no region thresholds to inherit: the training corpus was run "
                    "without --localise-limit, and tuning them here would use the "
                    "external data as a validation set"
                ),
            }
        )
        if localisation.get("available"):
            print(
                f"  {name} localisation: tiled Dice "
                f"{localisation['tiled']['dice']:.3f} IoU "
                f"{localisation['tiled']['iou']:.3f}",
                flush=True,
            )

        report["datasets"][name] = {
            "available": True,
            "images": int(len(frame)),
            "forged": int((y == 1).sum()),
            "authentic": int((y == 0).sum()),
            "scenes": int(frame["source_id"].nunique()),
            # Effective sample size is the scene count, not the image count, once
            # a corpus ships dozens of variants per photograph.
            "variants_per_scene_cap": int(variants_per_scene),
            "masks_available": int(
                (frame.loc[frame["label"] == 1, "mask_path"].astype(str) != "").sum()
            ),
            "image_level": metrics,
            "localisation": localisation,
            "by_tamper_type": by_family,
        }

    available = {
        key: value for key, value in report["datasets"].items() if value.get("available")
    }
    if available:
        report["summary"] = {
            "mean_roc_auc": float(
                np.mean([value["image_level"]["roc_auc"] for value in available.values()])
            ),
            "mean_balanced_accuracy": float(
                np.mean(
                    [value["image_level"]["balanced_accuracy"] for value in available.values()]
                )
            ),
            "mean_tiled_dice": float(
                np.mean(
                    [
                        value["localisation"]["tiled"]["dice"]
                        for value in available.values()
                        if value["localisation"].get("available")
                    ]
                )
                if any(
                    value["localisation"].get("available") for value in available.values()
                )
                else None,
            ),
        }
    return report
