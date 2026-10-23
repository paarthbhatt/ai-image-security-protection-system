"""End-to-end command for the current project milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dataset import build_benchmark
from protection import run_protection_checks
from train import train_and_evaluate
from visualize import generate_figures, write_presentation_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build, train, evaluate, and visualize the image security pilot.")
    parser.add_argument("--source-images", type=int, default=500, help="Number of source images to use (500 is a fast presentation pilot).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Rebuild the benchmark and overwrite generated artifacts.")
    args = parser.parse_args()

    raw_root = PROJECT_ROOT / "data" / "raw"
    processed_root = PROJECT_ROOT / "data" / "processed"
    model_root = PROJECT_ROOT / "models"
    report_root = PROJECT_ROOT / "reports"
    figure_root = PROJECT_ROOT / "figures"
    presentation_root = PROJECT_ROOT / "presentation"

    frame = build_benchmark(
        raw_root=raw_root,
        processed_root=processed_root,
        source_images=args.source_images,
        seed=args.seed,
        force=args.force,
    )
    print(f"Benchmark ready: {len(frame):,} image rows, {frame['source_id'].nunique():,} source groups")
    model, results, _ = train_and_evaluate(
        frame=frame,
        processed_root=processed_root,
        model_root=model_root,
        report_root=report_root,
        seed=args.seed,
    )
    first_test_path = frame[frame["split"] == "test"].iloc[0]["image_path"]
    protection = run_protection_checks(first_test_path, report_root / "protection_artifacts")
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
    print("\nCompleted.")
    print(f"Test ROC-AUC: {results['test']['roc_auc']:.3f}")
    print(f"Test F1:      {results['test']['f1']:.3f}")
    print(f"Figures:      {len(figures)} files in {figure_root}")
    print(f"Dashboard:    {report_root / 'current_status.html'}")


if __name__ == "__main__":
    main()
