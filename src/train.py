"""Train and evaluate the lightweight forensic baseline."""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, extract_frame, extract_features


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
    candidates = np.linspace(0.05, 0.95, 181)
    scores = [balanced_accuracy_score(y_true, probabilities >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def _postprocess(image: Image.Image, condition: str) -> Image.Image:
    if condition == "clean":
        return image.convert("RGB")
    if condition.startswith("jpeg_q"):
        quality = int(condition.removeprefix("jpeg_q"))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if condition == "resize_16":
        return image.resize((16, 16), Image.Resampling.BILINEAR).resize((32, 32), Image.Resampling.BILINEAR)
    if condition == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.0)).convert("RGB")
    raise ValueError(condition)


def _robustness_table(model, test_frame: pd.DataFrame, limit: int = 300) -> dict[str, dict[str, float]]:
    conditions = ["clean", "jpeg_q50", "jpeg_q20", "resize_16", "blur"]
    subset = test_frame.sample(n=min(limit, len(test_frame)), random_state=123)
    output: dict[str, dict[str, float]] = {}
    for condition in conditions:
        vectors = []
        labels = []
        for _, row in subset.iterrows():
            image = Image.open(row["image_path"]).convert("RGB")
            processed = np.asarray(_postprocess(image, condition), dtype=np.uint8)
            vectors.append(extract_features(processed))
            labels.append(int(row["label"]))
        probabilities = model.predict_proba(np.vstack(vectors))[:, 1]
        threshold = 0.5
        output[condition] = {
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, probabilities >= threshold)),
            "f1": float(f1_score(labels, probabilities >= threshold, zero_division=0)),
        }
    return output


def train_and_evaluate(
    frame: pd.DataFrame,
    processed_root: Path,
    model_root: Path,
    report_root: Path,
    seed: int = 42,
) -> tuple[Pipeline, dict[str, object], list[str]]:
    """Extract features, train the detector, evaluate it, and persist evidence."""
    processed_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    feature_path = processed_root / "features.npz"
    print("Extracting image-forensic features ...")
    X, feature_names = extract_frame(frame, progress_every=max(0, len(frame) // 5))
    y = frame["label"].to_numpy(dtype=np.int64)
    split = frame["split"].to_numpy(dtype=str)
    tamper_type = frame["tamper_type"].to_numpy(dtype=str)
    np.savez_compressed(
        feature_path,
        X=X,
        y=y,
        split=split,
        tamper_type=tamper_type,
        sample_id=frame["sample_id"].to_numpy(dtype=str),
        feature_names=np.asarray(feature_names, dtype=str),
    )

    train_mask = split == "train"
    val_mask = split == "validation"
    test_mask = split == "test"
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(X[train_mask], y[train_mask])
    val_probabilities = model.predict_proba(X[val_mask])[:, 1]
    threshold = _select_threshold(y[val_mask], val_probabilities)
    test_probabilities = model.predict_proba(X[test_mask])[:, 1]
    test_frame = frame.loc[test_mask].reset_index(drop=True)
    test_result = _metrics(y[test_mask], test_probabilities, threshold)
    validation_result = _metrics(y[val_mask], val_probabilities, threshold)
    majority_probability = np.full(np.sum(test_mask), float(np.mean(y[train_mask])))
    majority_result = _metrics(y[test_mask], majority_probability, 0.5)

    per_tamper: dict[str, dict[str, float]] = {}
    for name in sorted(set(tamper_type[test_mask])):
        if name == "authentic":
            continue
        mask = tamper_type[test_mask] == name
        if not np.any(mask):
            continue
        per_tamper[name] = {
            "count": int(mask.sum()),
            "recall": float(recall_score(np.ones(mask.sum()), test_probabilities[mask] >= threshold, zero_division=0)),
            "mean_risk_score": float(test_probabilities[mask].mean()),
        }

    fpr, tpr, roc_thresholds = roc_curve(y[test_mask], test_probabilities)
    precision, recall, pr_thresholds = precision_recall_curve(y[test_mask], test_probabilities)
    coefficients = model.named_steps["classifier"].coef_[0]
    metadata_path = processed_root / "dataset_metadata.txt"
    metadata_text = metadata_path.read_text(encoding="utf-8") if metadata_path.exists() else ""
    source_name = "configured pilot source"
    for line in metadata_text.splitlines():
        if line.startswith("Actual source used:"):
            source_name = line.split(":", 1)[1].strip()
            break
    feature_coefficients = sorted(
        ({"feature": name, "coefficient": float(value), "absolute": float(abs(value))}
         for name, value in zip(feature_names, coefficients)),
        key=lambda item: item["absolute"],
        reverse=True,
    )
    results: dict[str, object] = {
        "experiment": {
            "seed": seed,
            "source_groups": int(frame["source_id"].nunique()),
            "image_instances": int(len(frame)),
            "train_instances": int(train_mask.sum()),
            "validation_instances": int(val_mask.sum()),
            "test_instances": int(test_mask.sum()),
            "feature_count": len(feature_names),
            "model": "StandardScaler + balanced logistic regression",
            "data_source": source_name,
            "note": "Synthetic manipulations; source-safe split. See dataset_metadata.txt for provenance.",
        },
        "majority_baseline": majority_result,
        "validation": validation_result,
        "test": test_result,
        "per_tamper_type": per_tamper,
        "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
        "pr_curve": {"precision": precision, "recall": recall, "thresholds": pr_thresholds},
        "feature_coefficients": feature_coefficients,
        "robustness": _robustness_table(model, test_frame),
    }
    results = _json_ready(results)
    joblib.dump(
        {"model": model, "feature_names": feature_names, "threshold": threshold, "seed": seed},
        model_root / "forensic_detector.joblib",
    )
    (report_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return model, results, feature_names
