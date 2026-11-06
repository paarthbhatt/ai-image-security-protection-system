"""A fine-tuned ResNet-18 baseline, so the hand-crafted evidence has a competitor.

Why this module exists
----------------------
The project's headline number is a gradient-boosted classifier over 143 forensic
statistics. That is defensible only if it is compared, on the same rows, with the
kind of model the literature actually reports - and the base paper of this
project (Yadav & Vishwakarma, 2024, Table 9) reports a ResNet-CNN at CASIA
accuracy 0.8851, precision 0.8123, recall 0.9415, F1 0.8664, AUC 0.9596, MCC
0.7744. Without a deep baseline the claim "our feature set is competitive" has no
adversary in the repository.

What is shared with the classical pipeline, deliberately
--------------------------------------------------------
*   the split, read from the same benchmark CSV the pipeline wrote, so "test"
    means one thing across both models;
*   the decision rule, ``train.select_threshold``, fitted on validation balanced
    accuracy and never on test;
*   the metric definitions, ``train.evaluate_probabilities``, which is the whole
    six-column table of the base paper in one code path;
*   family stratification, ``cross_dataset.by_tamper_type``.

What differs, stated rather than hidden
---------------------------------------
*   Training rows are the plain split. The classical model additionally sees six
    JPEG-recompression copies of its training half; giving the network the same
    treatment means decoding and re-encoding 43,000 images, so this baseline is
    the weaker of the two on augmentation and the comparison should be read that
    way.
*   The base paper augments with horizontal flip, vertical flip and rotation
    through 180 degrees. Flips are kept; rotation is not, because resampling
    pixels is exactly the evidence a noise and DCT detector is asked to read.
*   Images are resized to a fixed square. An 8x8 JPEG block grid does not survive
    that intact, which is a real limitation of transferring a network trained this
    way onto compression forensics, and one reason to look at these numbers beside
    the feature-based ones rather than instead of them.

Hardware honesty
----------------
The GPU in this machine is an RTX 2050 with about 3.7 GiB usable memory, so
automatic mixed precision, a small batch, and a documented halve-the-batch retry
are not optimizations here, they are the condition of running at all. Peak
allocated bytes are reported in the output so the claim can be checked.

Usage
-----
    python src/deep_detector.py --benchmark data/processed/benchmark_casia2.csv
    python src/deep_detector.py --epochs 12 --batch-size 16
    python src/deep_detector.py --architecture resnet50   # closer to the paper's row
    python src/deep_detector.py --cross-dataset comofod
    python src/deep_detector.py --limit 32 --epochs 1     # smoke run: does it work,
                                                          # not how good is it
    python src/deep_detector.py --dry-run                 # two CPU steps, no data
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

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
        description="Fine-tune a ResNet-18 on an existing benchmark split and score it "
        "with the project's own metric block.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", default="data/processed/benchmark_casia2.csv",
                        help="Benchmark CSV written by run_pipeline.py; it carries the split.")
    parser.add_argument("--out", default="reports/arms/deep_resnet",
                        help="Directory for the JSON report and the weights.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Start here; the run halves it on an out-of-memory error and says so.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--architecture", default="resnet18", choices=["resnet18", "resnet50"],
                        help="resnet50 is the closer read of the base paper's ResNet-CNN row "
                             "and costs roughly three times the memory and the wall clock.")
    parser.add_argument("--weights", default="imagenet", choices=["imagenet", "scratch"],
                        help="Scratch is a different claim; do not swap it in to avoid a download.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap rows per fold, for a smoke run. A capped run answers 'does it "
                             "work', not 'how good is it', and the report records the cap.")
    parser.add_argument("--cross-dataset", default="",
                        help="Optional corpus name under data/real to score zero-shot with the "
                             "inherited threshold, the same contract the feature pipeline uses.")
    parser.add_argument("--cross-variants", type=int, default=4,
                        help="Variants per scene for a zero-shot corpus; see run_pipeline.")
    parser.add_argument("--cpu", action="store_true",
                        help="Allow a CPU run. Without this flag a missing GPU is fatal, because "
                             "a CPU 'baseline' number would be a different experiment.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Two optimisation steps on synthetic tensors; no images are read.")
    return parser.parse_args()


def _absolutise(value: object) -> str:
    text = str(value)
    if not text:
        return ""
    path = Path(text)
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def resolve_paths(frame: pd.DataFrame) -> pd.DataFrame:
    """Benchmark CSVs store repository-relative paths; make them usable here.

    ``notna``-then-replace rather than ``fillna("")``: a corpus with no masks at
    all gives a ``mask_path`` column that pandas reads as float64, and filling a
    float column with a string is a Pandas4Warning now and an error later. The
    warning came out of the test run, not out of a code review.
    """
    frame = frame.copy()
    for column in ("image_path", "mask_path"):
        if column not in frame:
            continue
        values = frame[column].astype("object")
        frame[column] = values.where(values.notna(), "").map(_absolutise)
    return frame


def cap_balanced(part: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Take a smoke-test subset from both classes, not the first ``limit`` rows.

    The benchmark CSV is written ordered by label, so a positional ``head``
    returned 24 authentic images and nothing else: the fold check below would
    then kill the run for the right reason, and ``--limit`` would simply be a
    flag that never works.
    """
    per_class = max(1, limit // 2)
    return part.groupby("label", sort=False).head(per_class).reset_index(drop=True)


def load_split(benchmark: Path, limit: int = 0) -> dict[str, pd.DataFrame]:
    """Fold the CSV by its recorded ``split`` column, refusing a broken fold."""
    frame = resolve_paths(pd.read_csv(benchmark))
    missing = {"split", "label", "image_path"} - set(frame.columns)
    if missing:
        raise SystemExit(f"{benchmark} is missing columns {sorted(missing)}")
    folds = {
        name: frame.loc[frame["split"] == name]
        for name in ("train", "validation", "test")
    }
    for name, part in folds.items():
        if part.empty:
            raise SystemExit(f"{benchmark} has no '{name}' rows; run the pipeline first")
        if limit:
            folds[name] = cap_balanced(part, limit)
    for name, part in folds.items():
        counts = part["label"].value_counts().to_dict()
        if len(counts) < 2:
            raise SystemExit(f"the '{name}' fold holds one class only: {counts}")
    return folds


class ForgeryImages(Dataset):
    """Decodes one image per row into a normalised 3-channel tensor.

    Deliberately lazy: 10,246 CASIA images at 224x224 float32 would be several
    gigabytes of RAM held in a tensor, which is the wrong budget on this machine.
    """

    def __init__(self, frame: pd.DataFrame, image_size: int, augment: bool) -> None:
        self.paths = [str(path) for path in frame["image_path"]]
        self.labels = pd.DataFrame(frame)["label"].astype(int).to_numpy()
        size = (image_size, image_size)
        steps: list[object] = []
        if augment:
            # Flips only: see the module docstring on why rotation is not here.
            steps.append(transforms.RandomHorizontalFlip())
            steps.append(transforms.RandomVerticalFlip())
        steps += [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        self.transform = transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
        with Image.open(self.paths[index]) as handle:
            image = handle.convert("RGB")
        return self.transform(image), float(self.labels[index])


def _trunks() -> dict[str, tuple]:
    """The architectures this script can build, each with its weight enum."""
    from torchvision import models

    return {
        "resnet18": (models.resnet18, models.ResNet18_Weights),
        "resnet50": (models.resnet50, models.ResNet50_Weights),
    }


def pick_checkpoint(architecture: str, weights: str) -> tuple[object, str]:
    """Resolve a checkpoint against the torchvision that is actually installed.

    torchvision 0.29.2 in this environment exposes ``IMAGENET1K_V2`` for resnet50
    and ``IMAGENET1K_V1`` only for resnet18, which is not what the usual examples
    assume. Writing either name by hand is a guess about someone else's install,
    and the guess fails at the worst possible moment: after the data is loaded,
    before any epoch has run. The returned tag is what the report records, so
    "pretrained on ImageNet" is never a bare claim here.
    """
    trunks = _trunks()
    if architecture not in trunks:
        raise SystemExit(f"unsupported architecture {architecture!r}")
    _, weight_enum = trunks[architecture]
    if weights == "scratch":
        return None, f"{architecture}:scratch"
    for name in ("IMAGENET1K_V2", "IMAGENET1K_V1"):
        option = getattr(weight_enum, name, None)
        if option is not None:
            return option, f"{architecture}:{name}"
    raise SystemExit(
        f"this torchvision build has no ImageNet checkpoint for {architecture}; "
        "refusing to fall back to scratch silently, because that is a different "
        "experiment"
    )


def build_model(architecture: str, weights: str, device: torch.device):
    """A trunk with a one-logit head, so the score is a probability via sigmoid."""
    checkpoint, tag = pick_checkpoint(architecture, weights)
    network = _trunks()[architecture][0](weights=checkpoint)
    network.fc = torch.nn.Linear(network.fc.in_features, 1)
    return network.to(device), tag


def _peak_gib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / (1 << 30)


@torch.no_grad()
def predict(model, frame: pd.DataFrame, image_size: int, batch_size: int,
            device: torch.device, workers: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(
        ForgeryImages(frame, image_size, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        scores.append(torch.sigmoid(logits.float()).view(-1).cpu().numpy())
        labels.append(target.numpy())
    return np.concatenate(labels), np.concatenate(scores)


def train_one_epoch(model, loader, optimizer, scaler, device) -> float:
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    total = 0.0
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = criterion(model(images).view(-1), target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach()) * len(target)
    return total / max(1, len(loader.dataset))


def dry_run(image_size: int) -> int:
    """Two synthetic steps: proves the plumbing without touching a dataset."""
    device = torch.device("cpu")
    model, _ = build_model("resnet18", "scratch", device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(enabled=False)
    images = torch.rand(2, 3, image_size, image_size)
    targets = torch.tensor([0.0, 1.0])
    for _ in range(2):
        optimizer.zero_grad()
        loss = torch.nn.BCEWithLogitsLoss()(model(images).view(-1), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    with torch.no_grad():
        probability = torch.sigmoid(model(images).view(-1))
    assert bool(torch.isfinite(loss)), "synthetic step produced a non-finite loss"
    assert bool(((probability >= 0.0) & (probability <= 1.0)).all()), "sigmoid left [0, 1]"
    print(f"dry run ok: loss {float(loss.detach()):.4f}, probabilities {probability.tolist()}")
    return 0


def main() -> int:
    from cross_dataset import by_tamper_type, cap_variants_per_scene
    from sources import DatasetUnavailable, load_real_forgery_dataset
    from train import evaluate_probabilities, select_threshold

    args = parse_args()
    if args.dry_run:
        return dry_run(args.image_size)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.cpu:
        device = torch.device("cpu")
        print("CPU run requested; a CPU baseline is not comparable with a GPU claim")
    else:
        raise SystemExit(
            "No CUDA device. Refusing to train silently on CPU: that would be a "
            "different experiment with the same name. Pass --cpu to mean it."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        print(f"device: {torch.cuda.get_device_name(0)}")

    benchmark = Path(args.benchmark)
    if not benchmark.is_absolute():
        benchmark = PROJECT_ROOT / benchmark
    folds = load_split(benchmark, args.limit)
    if args.limit:
        print(f"capped run: {args.limit} rows per fold - a smoke test, not a result")

    model, checkpoint_tag = build_model(args.architecture, args.weights, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    batch_size = args.batch_size
    reductions: list[str] = []
    started = time.time()
    best: dict[str, object] = {"balanced_accuracy": -1.0}
    best_state: dict[str, torch.Tensor] = {}

    for epoch in range(1, args.epochs + 1):
        loader = DataLoader(
            ForgeryImages(folds["train"], args.image_size, augment=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        while True:
            try:
                train_loss = train_one_epoch(model, loader, optimizer, scaler, device)
                break
            except torch.cuda.OutOfMemoryError:
                if batch_size <= 2:
                    raise
                batch_size //= 2
                reductions.append(f"epoch {epoch}: batch halved to {batch_size} (CUDA OOM)")
                print(reductions[-1], flush=True)
                torch.cuda.empty_cache()
                loader = DataLoader(
                    ForgeryImages(folds["train"], args.image_size, augment=True),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=args.workers,
                    pin_memory=device.type == "cuda",
                    generator=torch.Generator().manual_seed(args.seed + epoch),
                )
        scheduler.step()

        y_val, val_probability = predict(
            model, folds["validation"], args.image_size, batch_size, device, args.workers
        )
        val_metrics = evaluate_probabilities(y_val, val_probability, 0.5)
        print(
            f"epoch {epoch:>2}/{args.epochs}  loss {train_loss:.4f}  "
            f"val AUC {val_metrics['roc_auc']:.4f} "
            f"bal-ACC {val_metrics['balanced_accuracy']:.4f}  "
            f"peak {(_peak_gib(device)):.2f} GiB",
            flush=True,
        )
        if val_metrics["balanced_accuracy"] > float(best["balanced_accuracy"]):
            best = {
                "epoch": epoch,
                "balanced_accuracy": val_metrics["balanced_accuracy"],
                "roc_auc": val_metrics["roc_auc"],
            }
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        if val_metrics["roc_auc"] < 0.02 or val_metrics["roc_auc"] > 0.98:
            # An AUC pinned at an extreme on validation is not a strong model, it
            # is a collapsed one: every logit on one side of the threshold.
            print(
                f"  warning: validation AUC {val_metrics['roc_auc']:.4f} is degenerate; "
                "check the learning rate before believing any test number",
                flush=True,
            )

    if best_state:
        model.load_state_dict(best_state)

    val_rows, val_probability = predict(
        model, folds["validation"], args.image_size, batch_size, device, args.workers
    )
    threshold = select_threshold(val_rows, val_probability)

    y_test, test_probability = predict(
        model, folds["test"], args.image_size, batch_size, device, args.workers
    )
    test_metrics = evaluate_probabilities(y_test, test_probability, threshold)
    family_report = by_tamper_type(folds["test"], y_test, test_probability, threshold)

    report: dict[str, object] = {
        "model": {
            "architecture": f"{args.architecture} + one-logit head",
            "checkpoint": checkpoint_tag,
            "trainable_parameters": int(trainable),
            "image_size": int(args.image_size),
            "batch_size_final": int(batch_size),
            "batch_reductions": reductions,
            "epochs": int(args.epochs),
            "best_epoch": best.get("epoch"),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "augmentation": "horizontal and vertical flips; no rotation, no recompression copies",
            "amp": device.type == "cuda",
        },
        "provenance": {
            "benchmark_csv": _relative(benchmark),
            "rows_per_fold": {name: int(len(part)) for name, part in folds.items()},
            "labels_per_fold": {
                name: {str(k): int(v) for k, v in part["label"].value_counts().items()}
                for name, part in folds.items()
            },
            "row_cap": int(args.limit),
            "torch": torch.__version__,
            "torchvision": _torchvision_version(),
            "python": platform.python_version(),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "peak_vram_gib": round(_peak_gib(device), 3),
            "wall_seconds": round(time.time() - started, 1),
        },
        "validation": {
            "metrics": evaluate_probabilities(val_rows, val_probability, threshold),
            "threshold_selected": threshold,
        },
        "test": {"metrics": test_metrics, "per_family": family_report},
    }

    external = [name.strip() for name in args.cross_dataset.split(",") if name.strip()]
    if external:
        cross: dict[str, object] = {}
        for name in external:
            root = PROJECT_ROOT / "data" / "real" / name
            try:
                frame = load_real_forgery_dataset(name=name, root=root, seed=args.seed)
            except DatasetUnavailable as error:
                cross[name] = {"available": False, "reason": str(error)}
                print(f"zero-shot {name}: unavailable - {error}", flush=True)
                continue
            frame = cap_variants_per_scene(frame, args.cross_variants, args.seed)
            y_out, probability_out = predict(
                model, frame, args.image_size, batch_size, device, args.workers
            )
            metrics = evaluate_probabilities(y_out, probability_out, threshold)
            cross[name] = {
                "available": True,
                "images": int(len(frame)),
                "threshold_inherited_from_training_corpus": threshold,
                "metrics": metrics,
                "by_tamper_type": by_tamper_type(frame, y_out, probability_out, threshold),
            }
            print(
                f"zero-shot {name:10} ACC {metrics['accuracy']:.3f} "
                f"AUC {metrics['roc_auc']:.3f} MCC {metrics['mcc']:.3f} "
                f"({len(frame)} images, unseen)",
                flush=True,
            )
        report["cross_dataset"] = cross

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Tensors only, not a pickled trainer: the file can be read back without
    # executing anything but torch.load, and it names the layout it expects.
    torch.save(
        {
            "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "architecture": checkpoint_tag,
            "image_size": int(args.image_size),
            "threshold": threshold,
            "benchmark_csv": str(benchmark),
            "seed": int(args.seed),
        },
        out_dir / "resnet18.pt",
    )
    (out_dir / "deep_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    test = report["test"]["metrics"]
    print("\nCompleted.")
    print(f"Validation:   AUC {report['validation']['metrics']['roc_auc']:.4f}")
    print(f"Threshold:    {threshold:.3f} (validation only)")
    print(
        f"Test:         ACC {test['accuracy']:.4f} P {test['precision']:.4f} "
        f"R {test['recall']:.4f} F1 {test['f1']:.4f} AUC {test['roc_auc']:.4f} "
        f"MCC {test['mcc']:.4f}"
    )
    print(f"Peak VRAM:    {report['provenance']['peak_vram_gib']:.2f} GiB")
    print(f"Report:       {out_dir / 'deep_results.json'}")
    return 0


def _torchvision_version() -> str:
    import torchvision

    return torchvision.__version__


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return os.path.basename(str(path))


if __name__ == "__main__":
    raise SystemExit(main())
