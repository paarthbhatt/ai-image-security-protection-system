"""Invariants of the deep baseline (``src/deep_detector.py``).

The classical pipeline does not need PyTorch, so every test here skips instead of
going red when the import fails: a teammate's laptop without the CUDA install
should still produce a green suite, not a misleading one.

What is guarded, and why each one was worth writing:

*  ``--limit`` has to sample both classes. The real benchmark CSV is written
   ordered by label, so the obvious ``head(limit)`` returned 24 authentic images
   and no forgery, and the smoke-test flag simply never worked. Found by running
   it, not by reading it.
*  A fold that holds one class must be fatal rather than quietly scoreless.
*  The checkpoint name has to be resolved against the installed torchvision.
   This build has ``IMAGENET1K_V2`` for resnet50 and only ``IMAGENET1K_V1`` for
   resnet18; hard-coding either raised ``AttributeError`` after the data loaded.
*  ``predict`` returns probabilities in frame row order. Every family-level
   number in the report is computed by position against ``by_tamper_type``, so
   a reorder would mislabel families without changing any average.
"""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd
from PIL import Image
import numpy as np

try:
    import torch

    import deep_detector
    from deep_detector import PROJECT_ROOT

    HAS_TORCH = True
    TORCH_IMPORT_ERROR = ""
except Exception as error:  # torch/torchvision are optional for this suite
    HAS_TORCH = False
    TORCH_IMPORT_ERROR = f"{type(error).__name__}: {error}"

requires_torch = unittest.skipUnless(
    HAS_TORCH, f"PyTorch/torchvision not importable ({TORCH_IMPORT_ERROR})"
)


def _write_benchmark(directory: Path, rows: int = 8) -> Path:
    """A CSV shaped like the pipeline's, with real image files behind it."""
    images = directory / "img"
    images.mkdir()
    records = []
    for index in range(rows):
        path = images / f"{index:03d}.png"
        array = np.random.default_rng(index).integers(
            0, 255, size=(32, 40, 3), dtype=np.uint8
        )
        Image.fromarray(array).save(path)
        records.append(
            {
                "image_path": str(path),
                "mask_path": "",
                "label": index % 2,
                "split": ("train", "validation", "test")[index % 3],
                "tamper_type": "splicing" if index % 2 else "authentic",
            }
        )
    frame = pd.DataFrame(records)
    if frame["label"].nunique() > 1:
        # order it the way the real writer does: class first, then rows
        frame = frame.sort_values("label", kind="stable").reset_index(drop=True)
    path = directory / "benchmark_x.csv"
    frame.to_csv(path, index=False)
    return path


@requires_torch
class SmokeCapTest(unittest.TestCase):
    def setUp(self) -> None:
        # label-sorted, exactly as the pipeline writes it
        self.frame = pd.DataFrame(
            {
                "label": [0] * 10 + [1] * 10,
                "image_path": [f"p{i}.jpg" for i in range(20)],
            }
        )

    def test_a_positional_head_would_have_returned_one_class(self):
        """The defect this function exists to prevent, kept as an assertion."""
        self.assertEqual(set(self.frame.head(6)["label"]), {0})

    def test_the_cap_keeps_both_classes(self):
        capped = deep_detector.cap_balanced(self.frame, 6)
        self.assertEqual(sorted(capped["label"].unique()), [0, 1])
        self.assertEqual(len(capped), 6)
        self.assertEqual(capped["label"].value_counts().tolist(), [3, 3])

    def test_the_row_order_survives_the_cap(self):
        capped = deep_detector.cap_balanced(self.frame, 4)
        self.assertTrue(capped.index.is_monotonic_increasing)
        self.assertEqual(list(capped.index), [0, 1, 2, 3])


@requires_torch
class LoadSplitTest(unittest.TestCase):
    def test_folds_follow_the_split_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = _write_benchmark(Path(tmp), rows=9)
            folds = deep_detector.load_split(benchmark)
            self.assertEqual({name: len(part) for name, part in folds.items()},
                             {"train": 3, "validation": 3, "test": 3})

    def test_a_fold_missing_a_class_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = _write_benchmark(Path(tmp), rows=9)
            frame = pd.read_csv(benchmark)
            frame["label"] = 0
            frame.to_csv(benchmark, index=False)
            with self.assertRaises(SystemExit):
                deep_detector.load_split(benchmark)

    def test_an_absent_fold_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = _write_benchmark(Path(tmp), rows=9)
            frame = pd.read_csv(benchmark)
            frame["split"] = "train"
            frame.to_csv(benchmark, index=False)
            with self.assertRaises(SystemExit):
                deep_detector.load_split(benchmark)

    def test_a_missing_column_is_fatal_rather_than_a_keyerror_mid_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = _write_benchmark(Path(tmp), rows=9)
            frame = pd.read_csv(benchmark).drop(columns=["split"])
            frame.to_csv(benchmark, index=False)
            with self.assertRaises(SystemExit):
                deep_detector.load_split(benchmark)

    def test_capped_folds_keep_both_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = _write_benchmark(Path(tmp), rows=20)
            folds = deep_detector.load_split(benchmark, limit=4)
            for name, part in folds.items():
                self.assertEqual(part["label"].nunique(), 2, name)


@requires_torch
class PathResolutionTest(unittest.TestCase):
    def test_relative_paths_become_repository_absolute(self):
        frame = pd.DataFrame({"image_path": ["data/real/a.jpg"]})
        resolved = deep_detector.resolve_paths(frame)["image_path"].iloc[0]
        self.assertTrue(Path(resolved).is_absolute())
        self.assertTrue(str(resolved).startswith(str(PROJECT_ROOT)))

    def test_an_empty_mask_stays_empty_instead_of_becoming_the_string_nan(self):
        frame = pd.DataFrame({"image_path": ["a.jpg"], "mask_path": [None]})
        self.assertEqual(deep_detector.resolve_paths(frame)["mask_path"].iloc[0], "")

    def test_a_mask_column_that_is_empty_everywhere_does_not_become_a_float_column(self):
        """pandas reads an all-blank column as float64, and "" is not a float.

        This is the ordinary shape of a corpus with no ground truth at all, so
        the deep baseline used to hit it on every real run.
        """
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = _write_benchmark(Path(tmp), rows=9)
            frame = pd.read_csv(benchmark)
            frame["mask_path"] = np.nan
            frame.to_csv(benchmark, index=False)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                folds = deep_detector.load_split(benchmark)
            joined = pd.concat(folds.values())
            self.assertEqual(set(joined["mask_path"]), {""})
            # Not a dtype name check: pandas 3 infers StringDtype where this
            # assertion once expected object. The failure mode is a numeric or
            # NaN-holding column, and that is what stays fixed.
            self.assertFalse(pd.api.types.is_float_dtype(joined["mask_path"]))
            self.assertFalse(joined["mask_path"].isna().any())


@requires_torch
class CheckpointResolutionTest(unittest.TestCase):
    def test_scratch_is_named_as_scratch_and_loads_nothing(self):
        option, tag = deep_detector.pick_checkpoint("resnet18", "scratch")
        self.assertIsNone(option)
        self.assertEqual(tag, "resnet18:scratch")

    def test_the_resolved_checkpoint_exists_in_the_installed_torchvision(self):
        """The regression that cost a crash mid-launch.

        A name chosen here has to be an attribute of the enum in *this*
        environment, not in the one whose example code was copied.
        """
        from torchvision import models

        for architecture, weight_enum in (
            ("resnet18", models.ResNet18_Weights),
            ("resnet50", models.ResNet50_Weights),
        ):
            _option, tag = deep_detector.pick_checkpoint(architecture, "imagenet")
            self.assertTrue(tag.startswith(f"{architecture}:"))
            self.assertIn(tag.split(":", 1)[1], [w.name for w in weight_enum])

    def test_an_unknown_architecture_is_rejected_before_any_download(self):
        with self.assertRaises(SystemExit):
            deep_detector.pick_checkpoint("resnet9999", "imagenet")


@requires_torch
class HeadAndOrderingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.tag = deep_detector.build_model(
            "resnet18", "scratch", torch.device("cpu")
        )

    def test_the_head_emits_one_logit(self):
        self.assertEqual(self.model.fc.out_features, 1)
        with torch.no_grad():
            logit = self.model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(logit.shape), (2, 1))

    def test_predict_returns_probabilities_and_frame_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = _write_benchmark(Path(tmp), rows=9)
            folds = deep_detector.load_split(benchmark)
            frame = pd.concat(folds.values(), ignore_index=True)
            labels, probabilities = deep_detector.predict(
                self.model, frame, 64, 4, torch.device("cpu"), 0
            )
            self.assertEqual(len(labels), len(frame))
            # position, not distribution: by_tamper_type reads the frame rows
            self.assertEqual(list(labels), list(frame["label"].astype(float)))
            self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())

    def test_the_checkpoint_tag_reaches_the_saved_payload(self):
        """A weight file that does not say which weights it used is unverifiable."""
        self.assertTrue(self.tag.startswith("resnet18:"))


if __name__ == "__main__":
    unittest.main()
