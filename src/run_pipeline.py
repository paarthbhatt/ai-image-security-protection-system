"""End-to-end command: build a benchmark, train, evaluate, and visualise.

Examples
--------
Synthetic benchmark on real natural images::

    python src/run_pipeline.py --dataset cifar10 --source-images 3000

A real forged-image benchmark placed under ``data/real/casia2``::

    python src/run_pipeline.py --real-dataset casia2

Repeat over three seeds and aggregate::

    python src/run_pipeline.py --dataset cifar10 --seeds 42,7,13
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dataset import MANIPULATION_FAMILIES, ROBUSTNESS_CONDITIONS, build_benchmark, store_relative
from protection import run_protection_checks
from sources import DatasetUnavailable, load_real_forgery_dataset
from train import train_and_evaluate
from visualize import generate_figures, write_presentation_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build, train, evaluate, and visualise the image-integrity screener.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "digits"],
                        help="Authentic-image corpus for the synthetic benchmark.")
    parser.add_argument("--source-images", type=int, default=3000,
                        help="Number of source images taken from the corpus.")
    parser.add_argument("--real-dataset", default=None,
                        help="Name of a real forged-image benchmark under --real-root "
                             "(e.g. casia2, comofod, coverage). Skips synthesis entirely.")
    parser.add_argument("--real-root", default="data/real",
                        help="Directory holding manually downloaded real datasets.")
    parser.add_argument("--families", default=",".join(MANIPULATION_FAMILIES),
                        help="Comma-separated manipulation families to synthesise.")
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated seeds; the first produces the full artifact set.")
    parser.add_argument("--allow-fallback", action="store_true",
                        help="Permit the digits fallback if CIFAR-10 is unavailable. "
                             "Off by default so a broken download cannot pass silently.")
    parser.add_argument("--force-download", action="store_true", help="Re-download CIFAR-10.")
    parser.add_argument("--force", action="store_true", help="Rebuild the benchmark from scratch.")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip the post-processing sweep.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable post-processing augmentation of the training split. "
                             "Augmentation keeps the detector from collapsing under JPEG re-saving.")
    parser.add_argument("--augment-fraction", type=float, default=0.5,
                        help="Fraction of training rows re-used per augmentation condition.")
    parser.add_argument("--recompute-features", action="store_true",
                        help="Ignore cached feature matrices and extract from scratch.")
    return parser.parse_args()


def augmentation_conditions(args: argparse.Namespace) -> tuple[str, ...]:
    """Conditions used to harden the training set (never derived from test data)."""
    if args.no_augment:
        return ()
    return ("jpeg_q70", "jpeg_q30", "noise", "blur")


def build_frame(args: argparse.Namespace, seed: int, processed_root: Path | None = None):
    processed_root = processed_root or (PROJECT_ROOT / "data" / "processed")
    processed_root.mkdir(parents=True, exist_ok=True)
    if args.real_dataset:
        root = (PROJECT_ROOT / args.real_root / args.real_dataset).resolve()
        print(f"Loading real benchmark {args.real_dataset!r} from {root} ...")
        frame = load_real_forgery_dataset(
            name=args.real_dataset,
            root=root,
            mask_root=root,
            seed=seed,
        )
        store_relative(frame, PROJECT_ROOT).to_csv(
            processed_root / f"benchmark_{args.real_dataset}.csv", index=False
        )
        return frame

    families = tuple(item.strip() for item in args.families.split(",") if item.strip())
    return build_benchmark(
        raw_root=PROJECT_ROOT / "data" / "raw",
        processed_root=processed_root,
        source_images=args.source_images,
        seed=seed,
        force=args.force or seed != int(args.seeds.split(",")[0]),
        dataset=args.dataset,
        allow_fallback=args.allow_fallback,
        families=families,
        force_download=args.force_download,
    )


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    primary_seed = seeds[0]

    model_root = PROJECT_ROOT / "models"
    report_root = PROJECT_ROOT / "reports"
    figure_root = PROJECT_ROOT / "figures"
    presentation_root = PROJECT_ROOT / "presentation"
    conditions = () if args.skip_robustness else ROBUSTNESS_CONDITIONS
    augment = augmentation_conditions(args)

    frame = build_frame(args, primary_seed)
    print(
        f"Benchmark ready: {len(frame):,} rows, "
        f"{frame['source_id'].nunique():,} source groups, corpus={frame['dataset'].iloc[0]}"
    )
    print("Split sizes:", frame["split"].value_counts().to_dict())
    print("Labels:", frame["label"].value_counts().to_dict())

    model, results, _ = train_and_evaluate(
        frame=frame,
        processed_root=PROJECT_ROOT / "data" / "processed",
        model_root=model_root,
        report_root=report_root,
        seed=primary_seed,
        conditions=conditions,
        augment_conditions=augment,
        augmentation_fraction=args.augment_fraction,
        recompute_features=args.recompute_features or args.force,
    )

    if len(seeds) > 1:
        results["multi_seed"] = _run_extra_seeds(args, seeds[1:], model_root, report_root)

    first_test_path = frame[frame["split"] == "test"].iloc[0]["image_path"]
    protection = run_protection_checks(
        first_test_path, report_root / "protection_artifacts", root=PROJECT_ROOT
    )
    figures = generate_figures(frame, results, figure_root)
    write_presentation_artifacts(
        frame=frame,
        results=results,
        protection=protection,
        figure_dir=figure_root,
        report_dir=report_root,
        presentation_dir=presentation_root,
    )
    results["protection"] = protection
    (report_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    test = results["test"]
    low, high = results["confidence_intervals"]["test_roc_auc"]
    print("\nCompleted.")
    print(f"Model:        {results['experiment']['selected_model']}")
    print(f"Augmented:    {results['experiment']['augmentation']['extra_training_rows']:,} extra training rows")
    print(f"Test ROC-AUC: {test['roc_auc']:.3f}  (95% CI {low:.3f}-{high:.3f})")
    print(f"Test F1:      {test['f1']:.3f}")
    print(f"Leakage:      {'passed' if results['group_leakage_check']['passed'] else 'FAILED'}")
    print(f"Figures:      {len(figures)} files in {figure_root}")
    print(f"Dashboard:    {report_root / 'current_status.html'}")


def _run_extra_seeds(args: argparse.Namespace, seeds: list[int], model_root: Path, report_root: Path) -> dict:
    """Re-run the pipeline for additional seeds; keep metrics only.

    Each seed writes into its own ``data/processed/seed_<n>`` directory so the
    primary run's benchmark, features and figures are never overwritten.
    """
    summaries: dict[str, dict[str, float]] = {}
    for seed in seeds:
        print(f"\n=== Additional seed {seed} ===")
        seed_root = PROJECT_ROOT / "data" / "processed" / f"seed_{seed}"
        try:
            frame = build_frame(args, seed, processed_root=seed_root)
            _, seed_results, _ = train_and_evaluate(
                frame=frame,
                processed_root=seed_root,
                model_root=seed_root / "models",
                report_root=seed_root / "reports",
                seed=seed,
                run_robustness=False,
                augment_conditions=augmentation_conditions(args),
                augmentation_fraction=args.augment_fraction,
            )
            summaries[str(seed)] = {
                "roc_auc": seed_results["test"]["roc_auc"],
                "pr_auc": seed_results["test"]["pr_auc"],
                "f1": seed_results["test"]["f1"],
                "selected_model": seed_results["experiment"]["selected_model"],
            }
        except Exception as exc:  # noqa: BLE001 - one bad seed must not lose the others
            print(f"Seed {seed} failed: {exc}")
            summaries[str(seed)] = {"error": str(exc)}  # type: ignore[assignment]

    valid = [entry["roc_auc"] for entry in summaries.values() if "roc_auc" in entry]
    if valid:
        summaries["mean_roc_auc"] = float(np.mean(valid))  # type: ignore[assignment]
        summaries["std_roc_auc"] = float(np.std(valid))  # type: ignore[assignment]
    (report_root / "multiseed.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


if __name__ == "__main__":
    try:
        main()
    except DatasetUnavailable as error:
        raise SystemExit(
            f"\nDataset unavailable: {error}\n"
            "Nothing was trained. Fix the dataset or pass --allow-fallback explicitly."
        ) from error
