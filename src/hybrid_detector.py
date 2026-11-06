"""Model 4: Hybrid Forensic Detector.

Combines deep features from the fine-tuned ResNet-18 backbone with the 143
handcrafted forensic features, testing whether classical forensic clues still
provide complementary evidence on top of learned representations.

Why this module exists
----------------------
In Yadav & Vishwakarma (2024), their proposed architecture is a two-branch
design: one spatial branch and one compression-artifact branch.
Our 143 forensic features already explicitly model compression artifacts (JPEG
DCT profiles, ELA, chroma shifts) and pixel residuals (SRM, noise, copy-move).
By concatenating the 512-dimensional penultimate representations from the fine-tuned
ResNet-18 with the 143 forensic features, we form a 655-dimensional hybrid representation
and fit a tree ensemble.

Protocol contracts:
- Uses the identical benchmark split (data/processed/benchmark_casia2.csv).
- Threshold chosen on validation split only.
- Reports the same 6 metrics: ACC, P, R, F1, AUC, MCC.
- Evaluates zero-shot on external corpora (CoMoFoD) using the inherited threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract deep embeddings from Model 3, concatenate with 143 forensic features, and train Model 4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", default="data/processed/benchmark_casia2.csv",
                        help="Benchmark CSV carrying the frozen split.")
    parser.add_argument("--weights", default="reports/arms/deep_resnet/resnet18.pt",
                        help="Checkpoint from Model 3.")
    parser.add_argument("--forensic-cache", default="data/processed/features.npz",
                        help="Pre-extracted 143 forensic features for CASIA.")
    parser.add_argument("--out", default="reports/arms/hybrid",
                        help="Output directory for reports and metrics.")
    parser.add_argument("--model-out", default="models/arms/hybrid",
                        help="Directory to save the fitted hybrid model.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cross-dataset", default="comofod",
                        help="Comma-separated external datasets to score zero-shot.")
    parser.add_argument("--cross-variants", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="Row cap per fold for smoke testing.")
    return parser.parse_args()


class SimpleImageDataset(Dataset):
    def __init__(self, paths: list[str], image_size: int) -> None:
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        with Image.open(path) as handle:
            img = handle.convert("RGB")
        return self.transform(img)


def load_feature_extractor(checkpoint_path: Path, device: torch.device):
    """Loads ResNet-18 checkpoint from Model 3 and strips the classification head."""
    from torchvision import models

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload["state_dict"] if "state_dict" in payload else payload

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(state_dict)

    # Replace fc with Identity to output 512-dim embedding
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)


@torch.no_grad()
def extract_deep_embeddings(model, paths: list[str], image_size: int, batch_size: int,
                            device: torch.device, workers: int) -> np.ndarray:
    loader = DataLoader(
        SimpleImageDataset(paths, image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    embeddings: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            emb = model(batch)
        embeddings.append(emb.float().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def main() -> int:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from cross_dataset import by_tamper_type, cap_variants_per_scene, load_external_frame, _cached_matrix
    from deep_detector import resolve_paths
    from features import feature_names
    from train import evaluate_probabilities, model_zoo, select_threshold

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.is_absolute():
        benchmark_path = PROJECT_ROOT / benchmark_path

    ckpt_path = Path(args.weights)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    if not ckpt_path.exists():
        raise SystemExit(f"Model 3 checkpoint {ckpt_path} does not exist. Train Model 3 first.")

    forensic_cache_path = Path(args.forensic_cache)
    if not forensic_cache_path.is_absolute():
        forensic_cache_path = PROJECT_ROOT / forensic_cache_path

    if not forensic_cache_path.exists():
        raise SystemExit(f"Forensic cache {forensic_cache_path} missing. Run run_pipeline first.")

    print(f"Loading benchmark {benchmark_path} ...")
    frame = resolve_paths(pd.read_csv(benchmark_path))
    if args.limit > 0:
        per_class = max(1, args.limit // 2)
        frame = frame.groupby(["split", "label"], sort=False).head(per_class).reset_index(drop=True)
        print(f"Capped run: {len(frame)} total rows across splits.")

    print(f"Loading forensic features from {forensic_cache_path} ...")
    forensic_data = np.load(forensic_cache_path)
    X_forensic_full = forensic_data["X"]
    sample_ids_full = list(forensic_data["sample_id"])
    id_to_idx = {sid: i for i, sid in enumerate(sample_ids_full)}

    # Map rows
    row_indices = [id_to_idx[sid] for sid in frame["sample_id"] if sid in id_to_idx]
    if len(row_indices) != len(frame):
        raise SystemExit(f"Mismatch: frame has {len(frame)} rows, found {len(row_indices)} in forensic cache.")
    X_forensic = X_forensic_full[row_indices]

    print(f"Loading Model 3 backbone from {ckpt_path} ...")
    backbone = load_feature_extractor(ckpt_path, device)

    cache_dir = PROJECT_ROOT / "data" / "processed" / "arms" / "hybrid"
    cache_dir.mkdir(parents=True, exist_ok=True)
    deep_cache_file = cache_dir / f"deep_emb_casia_{len(frame)}.npy"

    if deep_cache_file.exists() and args.limit == 0:
        print(f"Loading cached deep embeddings from {deep_cache_file} ...")
        X_deep = np.load(deep_cache_file)
    else:
        print(f"Extracting 512-dim deep embeddings for {len(frame)} images ...")
        t0 = time.time()
        X_deep = extract_deep_embeddings(
            backbone, list(frame["image_path"]), args.image_size, args.batch_size, device, args.workers
        )
        print(f"Extracted deep embeddings in {time.time() - t0:.1f}s, shape: {X_deep.shape}")
        if args.limit == 0:
            np.save(deep_cache_file, X_deep)

    # Concatenate features
    X_hybrid = np.hstack([X_deep, X_forensic])
    print(f"Hybrid feature matrix shape: {X_hybrid.shape} (512 deep + {X_forensic.shape[1]} forensic)")

    y = frame["label"].to_numpy(dtype=np.int64)
    splits = frame["split"].to_numpy()

    train_mask = (splits == "train")
    val_mask = (splits == "validation")
    test_mask = (splits == "test")

    X_train, y_train = X_hybrid[train_mask], y[train_mask]
    X_val, y_val = X_hybrid[val_mask], y[val_mask]
    X_test, y_test = X_hybrid[test_mask], y[test_mask]

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # Fit models
    zoo = model_zoo(args.seed)
    candidates = {}
    print("\nFitting models on hybrid representations ...")
    for name in ["logistic_regression", "random_forest", "hist_gradient_boosting"]:
        t_fit = time.time()
        clf = zoo[name]
        clf.fit(X_train, y_train)
        val_prob = clf.predict_proba(X_val)[:, 1]
        val_auc = evaluate_probabilities(y_val, val_prob, 0.5)["roc_auc"]
        candidates[name] = {"model": clf, "val_auc": val_auc}
        print(f"  {name:25} fit in {time.time() - t_fit:.1f}s | Val ROC-AUC: {val_auc:.4f}")

    selected_name = max(candidates.keys(), key=lambda k: candidates[k]["val_auc"])
    selected_model = candidates[selected_name]["model"]
    print(f"\nSelected Model: {selected_name} (Val AUC: {candidates[selected_name]['val_auc']:.4f})")

    # Select threshold on validation
    val_prob = selected_model.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y_val, val_prob)
    val_metrics = evaluate_probabilities(y_val, val_prob, threshold)

    # Evaluate on held-out test
    test_prob = selected_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_probabilities(y_test, test_prob, threshold)
    family_report = by_tamper_type(frame.loc[test_mask], y_test, test_prob, threshold)

    print(f"Validation: AUC {val_metrics['roc_auc']:.4f} | Balanced ACC: {val_metrics['balanced_accuracy']:.4f}")
    print(f"Threshold:  {threshold:.3f} (selected on validation)")
    print(
        f"Test:       ACC {test_metrics['accuracy']:.4f} P {test_metrics['precision']:.4f} "
        f"R {test_metrics['recall']:.4f} F1 {test_metrics['f1']:.4f} "
        f"AUC {test_metrics['roc_auc']:.4f} MCC {test_metrics['mcc']:.4f}"
    )

    report = {
        "model": {
            "name": "Model 4: Hybrid (ResNet-18 + 143 Forensic Features)",
            "selected_classifier": selected_name,
            "deep_backbone": "resnet18 (ImageNet-1K fine-tuned)",
            "feature_dim": int(X_hybrid.shape[1]),
            "deep_dim": int(X_deep.shape[1]),
            "forensic_dim": int(X_forensic.shape[1]),
            "threshold": threshold,
        },
        "validation": val_metrics,
        "test": {"metrics": test_metrics, "per_family": family_report},
    }

    # Zero-shot cross-dataset evaluation
    external = [name.strip() for name in args.cross_dataset.split(",") if name.strip()]
    if external and "comofod" in external:
        print("\nEvaluating zero-shot transfer on CoMoFoD ...")
        comofod_frame = load_external_frame("comofod", PROJECT_ROOT / "data" / "real" / "comofod", variants_per_scene=args.cross_variants)
        comofod_forensic_path = PROJECT_ROOT / "data" / "processed" / "features_cross_comofod.npz"
        comofod_forensic = _cached_matrix(comofod_forensic_path, comofod_frame, feature_names())

        comofod_deep_cache = cache_dir / f"deep_emb_comofod_{len(comofod_frame)}.npy"
        if comofod_deep_cache.exists() and args.limit == 0:
            comofod_deep = np.load(comofod_deep_cache)
        else:
            print(f"Extracting deep embeddings on {len(comofod_frame)} CoMoFoD images ...")
            comofod_deep = extract_deep_embeddings(
                backbone, list(comofod_frame["image_path"]), args.image_size, args.batch_size, device, args.workers
            )
            if args.limit == 0:
                np.save(comofod_deep_cache, comofod_deep)

        X_comofod_hybrid = np.hstack([comofod_deep, comofod_forensic])
        y_comofod = comofod_frame["label"].to_numpy(dtype=np.int64)

        comofod_prob = selected_model.predict_proba(X_comofod_hybrid)[:, 1]
        comofod_metrics = evaluate_probabilities(y_comofod, comofod_prob, threshold)
        comofod_families = by_tamper_type(comofod_frame, y_comofod, comofod_prob, threshold)

        print(
            f"Zero-shot CoMoFoD: ACC {comofod_metrics['accuracy']:.4f} "
            f"AUC {comofod_metrics['roc_auc']:.4f} MCC {comofod_metrics['mcc']:.4f}"
        )
        report["cross_dataset"] = {
            "comofod": {
                "available": True,
                "images": len(comofod_frame),
                "threshold_inherited": threshold,
                "metrics": comofod_metrics,
                "by_tamper_type": comofod_families,
            }
        }

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_out)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"model": selected_model, "threshold": threshold, "classifier": selected_name, "feature_dim": X_hybrid.shape[1]},
        model_dir / "hybrid_detector.joblib"
    )
    (out_dir / "hybrid_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report to {out_dir / 'hybrid_results.json'}")
    print(f"Saved model to {model_dir / 'hybrid_detector.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

