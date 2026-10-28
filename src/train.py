"""Train and evaluate image-manipulation detectors.

The evaluation follows the protocol in ``docs_research_plan.md``:

* thresholds are chosen on the validation split, never on test;
* every model in the zoo is reported, so a weak baseline cannot be hidden;
* ROC-AUC and PR-AUC lead, because accuracy hides imbalance;
* bootstrap confidence intervals are attached to the headline numbers;
* robustness is measured on the *same* learned threshold, across JPEG, rescale,
  blur and noise conditions.

``HistGradientBoostingClassifier`` is the primary model: on tabular forensic
features it consistently beats a linear detector, while ``LogisticRegression``
is kept as the interpretable reference required by the research plan.
"""

from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import extract_frame, extract_paths
from features import feature_names as feature_frame_names
from features import FEATURE_VERSION
from sources import content_fingerprint

PRIMARY_MODEL = "hist_gradient_boosting"


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def model_zoo(seed: int) -> dict[str, object]:
    """Detector candidates spanning linear, bagged and boosted families."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000, class_weight="balanced", random_state=seed, solver="lbfgs"
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        PRIMARY_MODEL: HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def _select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Pick the threshold maximising balanced accuracy on validation only."""
    candidates = np.linspace(0.05, 0.95, 181)
    scores = [balanced_accuracy_score(y_true, probabilities >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def _bootstrap_ci(
    y_true: np.ndarray, probabilities: np.ndarray, metric, replicates: int = 400, seed: int = 7
) -> tuple[float, float]:
    """Percentile bootstrap interval for a metric that needs both classes."""
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(y_true)
    for _ in range(replicates):
        indices = rng.integers(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        values.append(float(metric(y_true[indices], probabilities[indices])))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _postprocess(image: Image.Image, condition: str, seed: int = 0) -> Image.Image:
    """Apply one post-processing condition, preserving image size."""
    if condition == "clean":
        return image.convert("RGB")
    if condition.startswith("jpeg_q"):
        quality = int(condition.removeprefix("jpeg_q"))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if condition == "resize_half":
        width, height = image.size
        small = image.resize((max(8, width // 2), max(8, height // 2)), Image.Resampling.BILINEAR)
        return small.resize((width, height), Image.Resampling.BILINEAR)
    if condition == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.0)).convert("RGB")
    if condition == "noise":
        rng = np.random.default_rng(seed)
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        array = np.clip(array + rng.normal(0.0, 6.0, array.shape), 0, 255)
        return Image.fromarray(array.astype(np.uint8))
    raise ValueError(f"Unknown robustness condition: {condition}")


def robustness_table(
    model,
    test_frame: pd.DataFrame,
    threshold: float,
    conditions: tuple[str, ...],
    limit: int = 400,
) -> dict[str, dict[str, float]]:
    """Re-score a test subset under each post-processing condition."""
    subset = test_frame.sample(n=min(limit, len(test_frame)), random_state=123)
    output: dict[str, dict[str, float]] = {}
    for condition in conditions:
        vectors = extract_paths(
            subset["image_path"],
            transform=lambda image, name=condition: _postprocess(image, name),
            stage=f"[{condition}]",
        )
        labels_array = subset["label"].to_numpy()
        probabilities = model.predict_proba(vectors)[:, 1]
        output[condition] = {
            "roc_auc": float(roc_auc_score(labels_array, probabilities)),
            "balanced_accuracy": float(balanced_accuracy_score(labels_array, probabilities >= threshold)),
            "f1": float(f1_score(labels_array, probabilities >= threshold, zero_division=0)),
        }
    return output


def _family_evaluation(
    model, subset: pd.DataFrame, X_subset: np.ndarray, threshold: float
) -> dict[str, dict[str, float]]:
    """Binary detection quality per forgery family, authentic images as negatives."""
    results: dict[str, dict[str, float]] = {}
    if subset.empty:
        return results
    authentic = (subset["label"] == 0).to_numpy()
    for family in sorted(set(subset["tamper_type"]) - {"authentic"}):
        forged = (subset["tamper_type"] == family).to_numpy()
        mask = forged | authentic
        if not forged.any() or not authentic.any():
            continue
        y = forged[mask].astype(int)
        probabilities = model.predict_proba(X_subset[mask])[:, 1]
        results[family] = {
            "count": int(forged.sum()),
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "pr_auc": float(average_precision_score(y, probabilities)),
            "recall": float(recall_score(y, probabilities >= threshold, zero_division=0)),
            "mean_risk_score": float(probabilities[y == 1].mean()),
        }
    return results


def _load_cached_matrix(
    path: Path, frame: pd.DataFrame, names: list[str]
) -> np.ndarray | None:
    """Reuse a stored feature matrix only if it describes these exact pixels.

    Keyed on the ordered ``sample_id`` list, the feature-name vector, the
    extractor version, and a digest of the image files. The digest matters
    because sample ids are positional: regenerating the benchmark with a
    different seed reuses every id while changing every image, so an id-only
    check would happily return rows extracted from the previous corpus.
    """
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            stored_ids = [str(value) for value in data["sample_id"]]
            stored_names = [str(value) for value in data["feature_names"]]
            stored_version = str(data["feature_version"]) if "feature_version" in data else ""
            stored_content = str(data["content_key"]) if "content_key" in data else ""
            matrix = data["X"]
    except Exception:  # noqa: BLE001 - an unreadable cache is just a cache miss
        return None
    if stored_version != FEATURE_VERSION:
        return None
    if stored_names != names or stored_ids != list(frame["sample_id"].astype(str)):
        return None
    if stored_content != content_fingerprint(frame["image_path"].astype(str)):
        return None
    return matrix


def _augment_training(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    conditions: tuple[str, ...],
    fraction: float,
    seed: int,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Expand the training split with post-processed copies of itself.

    A detector trained only on clean images learns their exact compression and
    noise signature, so re-saving at a different JPEG quality moves the whole
    feature distribution and the model collapses. Showing it those conditions
    during training is the standard remedy and costs no labelled data.

    Only training rows are touched; validation and test stay pristine.
    """
    if not conditions or fraction <= 0:
        return None

    cache_key = _frame_fingerprint(frame, train_mask, seed)
    if cache_path is not None and Path(cache_path).exists():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                if (
                    [str(value) for value in data["conditions"]] == list(conditions)
                    and str(data["frame_key"]) == cache_key
                ):
                    print(f"  reusing augmented features from {Path(cache_path).name}", flush=True)
                    return data["X"], data["y"]
        except Exception:  # noqa: BLE001 - rebuild on any mismatch
            pass
    train_rows = frame.loc[train_mask]
    if train_rows.empty:
        return None

    vectors: list[np.ndarray] = []
    labels: list[int] = []
    for index, condition in enumerate(conditions):
        subset = train_rows.sample(frac=min(1.0, fraction), random_state=seed + index)
        if subset.empty:
            continue
        vectors.append(
            extract_paths(
                subset["image_path"],
                transform=lambda image, name=condition: _postprocess(image, name),
                stage=f"[{condition}]",
            )
        )
        labels.append(subset["label"].to_numpy(dtype=np.int64))
        print(f"  augmented with {condition}: {len(subset)} rows", flush=True)

    if not vectors:
        return None
    X_aug = np.vstack(vectors)
    y_aug = np.concatenate(labels)
    if cache_path is not None:
        np.savez_compressed(
            cache_path,
            X=X_aug,
            y=y_aug,
            conditions=np.asarray(conditions, dtype=str),
            frame_key=np.asarray(cache_key),
        )
    return X_aug, y_aug


def _frame_fingerprint(frame: pd.DataFrame, train_mask: np.ndarray, seed: int) -> str:
    """Digest of the rows an augmented cache was built from, plus the seed.

    The seed is part of the key because the augmentation subset is drawn with
    ``seed + index``, and the content digest because regenerating the benchmark
    can reuse every ``sample_id`` while changing every pixel.
    """
    ids = "|".join(frame.loc[train_mask, "sample_id"].astype(str))
    digest = hashlib.sha256(ids.encode("utf-8"))
    digest.update(str(seed).encode("ascii"))
    digest.update(FEATURE_VERSION.encode("utf-8"))
    digest.update(content_fingerprint(frame.loc[train_mask, "image_path"].astype(str)).encode("ascii"))
    return digest.hexdigest()[:16]


def train_and_evaluate(
    frame: pd.DataFrame,
    processed_root: Path,
    model_root: Path,
    report_root: Path,
    seed: int = 42,
    conditions: tuple[str, ...] = (),
    run_robustness: bool = True,
    augment_conditions: tuple[str, ...] = (),
    augmentation_fraction: float = 0.5,
    recompute_features: bool = False,
) -> tuple[object, dict[str, object], list[str]]:
    """Extract features, train every candidate model, evaluate, and persist evidence."""
    processed_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    feature_names = feature_frame_names()

    print("Extracting forensic features ...", flush=True)
    feature_path = processed_root / "features.npz"
    cached_matrix = None if recompute_features else _load_cached_matrix(
        feature_path, frame, feature_names
    )
    if cached_matrix is not None:
        print(f"  reusing {feature_path.name} ({cached_matrix.shape[0]} rows, "
              f"{cached_matrix.shape[1]} features)", flush=True)
        X = cached_matrix
    else:
        X, feature_names = extract_frame(frame)
    y = frame["label"].to_numpy(dtype=np.int64)
    split = frame["split"].to_numpy(dtype=str)

    train_mask, val_mask, test_mask = split == "train", split == "validation", split == "test"
    if train_mask.sum() < 10 or val_mask.sum() < 4 or test_mask.sum() < 4:
        raise RuntimeError(
            f"Split too small for evaluation: train={train_mask.sum()}, "
            f"validation={val_mask.sum()}, test={test_mask.sum()}."
        )

    np.savez_compressed(
        processed_root / "features.npz",
        X=X, y=y, split=split,
        tamper_type=frame["tamper_type"].to_numpy(dtype=str),
        sample_id=frame["sample_id"].to_numpy(dtype=str),
        source_id=frame["source_id"].to_numpy(dtype=str),
        feature_names=np.asarray(feature_names, dtype=str),
        feature_version=np.asarray(FEATURE_VERSION),
        content_key=np.asarray(content_fingerprint(frame["image_path"].astype(str))),
    )

    comparison: dict[str, dict[str, object]] = {}
    fitted: dict[str, object] = {}
    thresholds: dict[str, float] = {}

    augmented = _augment_training(
        frame, train_mask, augment_conditions, augmentation_fraction, seed,
        cache_path=processed_root / "features_augmented.npz",
    )
    if augmented is not None:
        X_fit = np.vstack([X[train_mask], augmented[0]])
        y_fit = np.concatenate([y[train_mask], augmented[1]])
        print(f"  training rows: {len(y[train_mask])} clean + {len(augmented[1])} augmented", flush=True)
    else:
        X_fit, y_fit = X[train_mask], y[train_mask]

    for name, model in model_zoo(seed).items():
        print(f"Fitting {name} ...", flush=True)
        model.fit(X_fit, y_fit)
        validation_probabilities = model.predict_proba(X[val_mask])[:, 1]
        threshold = _select_threshold(y[val_mask], validation_probabilities)
        test_probabilities = model.predict_proba(X[test_mask])[:, 1]
        comparison[name] = {
            "validation": _metrics(y[val_mask], validation_probabilities, threshold),
            "test": _metrics(y[test_mask], test_probabilities, threshold),
        }
        fitted[name], thresholds[name] = model, threshold

    best_name = max(comparison, key=lambda key: comparison[key]["validation"]["roc_auc"])
    model = fitted[best_name]
    threshold = thresholds[best_name]
    test_probabilities = model.predict_proba(X[test_mask])[:, 1]
    validation_probabilities = model.predict_proba(X[val_mask])[:, 1]

    roc_low, roc_high = _bootstrap_ci(y[test_mask], test_probabilities, roc_auc_score)
    pr_low, pr_high = _bootstrap_ci(y[test_mask], test_probabilities, average_precision_score)
    f1_low, f1_high = _bootstrap_ci(
        y[test_mask], test_probabilities, lambda t, p: f1_score(t, p >= threshold, zero_division=0)
    )

    test_frame = frame.loc[test_mask].reset_index(drop=True)
    majority_probability = np.full(int(test_mask.sum()), float(np.mean(y[train_mask])))
    fpr, tpr, roc_thresholds = roc_curve(y[test_mask], test_probabilities)
    precision, recall, pr_thresholds = precision_recall_curve(y[test_mask], test_probabilities)

    feature_importances = _feature_importances(model, feature_names)
    metadata_path = processed_root / "dataset_metadata.txt"
    corpus_name = "unknown"
    if metadata_path.exists():
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("corpus="):
                corpus_name = line.split("=", 1)[1].strip()
                break

    split_leakage = _check_group_leakage(frame)
    results: dict[str, object] = {
        "experiment": {
            "seed": seed,
            "corpus": corpus_name,
            "datasets": sorted(frame["dataset"].unique().tolist()) if "dataset" in frame else [],
            "source_groups": int(frame["source_id"].nunique()),
            "image_instances": int(len(frame)),
            "train_instances": int(train_mask.sum()),
            "validation_instances": int(val_mask.sum()),
            "test_instances": int(test_mask.sum()),
            "feature_count": len(feature_names),
            "feature_groups": sorted({name.split("_")[0] for name in feature_names}),
            "selected_model": best_name,
            "models_compared": sorted(comparison),
            "augmentation": {
                "conditions": list(augment_conditions),
                "fraction": augmentation_fraction,
                "extra_training_rows": int(0 if augmented is None else len(augmented[1])),
            },
            "note": (
                "Model chosen on validation ROC-AUC. Ground-truth masks are stored "
                "per manipulated image; image-level screening only (no localization metric)."
            ),
        },
        "confidence_intervals": {
            "test_roc_auc": [roc_low, roc_high],
            "test_pr_auc": [pr_low, pr_high],
            "test_f1": [f1_low, f1_high],
            "method": "percentile bootstrap, 400 replicates",
        },
        "group_leakage_check": split_leakage,
        "model_comparison": comparison,
        "majority_baseline": _metrics(y[test_mask], majority_probability, 0.5),
        "validation": _metrics(y[val_mask], validation_probabilities, threshold),
        "test": _metrics(y[test_mask], test_probabilities, threshold),
        "per_family": _family_evaluation(model, test_frame, X[test_mask], threshold),
        "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
        "pr_curve": {"precision": precision, "recall": recall, "thresholds": pr_thresholds},
        "feature_importances": feature_importances,
        "robustness": (
            robustness_table(model, test_frame, threshold, conditions)
            if (run_robustness and conditions)
            else {}
        ),
    }
    results = _json_ready(results)

    payload = {
        "model": model,
        "model_name": best_name,
        "feature_names": feature_names,
        "threshold": threshold,
        "seed": seed,
        "corpus": corpus_name,
    }
    joblib.dump(payload, model_root / "forensic_detector.joblib")
    (report_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return model, results, feature_names


def _feature_importances(model, feature_names: list[str]) -> list[dict[str, object]]:
    """Model-agnostic ranking: linear coefficients or impurity importance."""
    classifier = model.named_steps["classifier"] if isinstance(model, Pipeline) else model
    values = None
    if hasattr(classifier, "coef_"):
        values = np.asarray(classifier.coef_)[0]
    elif hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_)
    if values is None or len(values) != len(feature_names):
        return []
    ranked = sorted(
        ({"feature": name, "value": float(value), "absolute": float(abs(value))}
         for name, value in zip(feature_names, values)),
        key=lambda item: item["absolute"],
        reverse=True,
    )
    return ranked[:40]


def _check_group_leakage(frame: pd.DataFrame) -> dict[str, object]:
    """Verify that no source group appears in more than one split."""
    crossing = frame.groupby("source_id")["split"].nunique()
    offenders = crossing[crossing > 1]
    return {
        "groups_in_multiple_splits": int(len(offenders)),
        "examples": offenders.index.tolist()[:5],
        "passed": bool(len(offenders) == 0),
    }
