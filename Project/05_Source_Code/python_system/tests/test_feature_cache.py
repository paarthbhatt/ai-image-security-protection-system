"""The feature cache is the cost model of this whole project, so its rules are tests.

Extraction dominates every runtime number here: roughly 12,600 corpus images plus
26,200 augmented training copies, and everything else is seconds. That makes the
cache's *validation* rules load-bearing, and a cache that silently misses is
invisible except in the wall clock.

Two behaviours are pinned, both measured before being written down:

*  A valid cache must prevent re-extraction - including across ``--seeds`` runs,
   where each seed writes into its own directory. It used to look for a cache
   only inside that directory, find nothing, and re-extract the entire corpus per
   seed. A variance pass cost an hour per seed, which is how it stayed undone.
*  A cache must not be trusted when the pixels, the feature layout, or the row
   identity changed. The check is positional on purpose, and one test below
   records that as the behaviour rather than a surprise.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests import fixtures

import sources
import train


class _CountingExtractor:
    """Stands in for ``train.extract_frame`` and records every call.

    It raises rather than returning a matrix, so a cache miss becomes a test
    failure with a message instead of a slow test that still passes.
    """

    def __init__(self, matrix: np.ndarray | None = None) -> None:
        self.calls = 0
        self.matrix = matrix

    def __call__(self, frame: pd.DataFrame):
        self.calls += 1
        if self.matrix is None:
            raise AssertionError("extract_frame ran: the feature cache was not reused")
        import features

        return self.matrix, features.feature_names()


def _run(frame: pd.DataFrame, processed_root: Path, seed: int = 42, **kwargs) -> dict:
    root = Path(processed_root)
    _, results, _ = train.train_and_evaluate(
        frame=frame,
        processed_root=root,
        model_root=root / "models",
        report_root=root / "reports",
        seed=seed,
        conditions=(),
        run_robustness=False,
        augment_conditions=(),
        localise_limit=0,
        **kwargs,
    )
    return results


class CacheFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        fixtures.build_fake_casia(cls.root / "casia2", scenes=8)
        cls.frame = sources.load_real_forgery_dataset(
            name="casia2",
            root=cls.root / "casia2",
            mask_root=cls.root / "casia2",
            seed=42,
        )
        if cls.frame["label"].nunique() < 2:
            raise unittest.SkipTest("fixture produced one class only")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def setUp(self) -> None:
        # every test gets its own artifact directory: the cache is the thing under
        # test, and a shared one would make the order of tests decide the result
        self.work = Path(tempfile.mkdtemp(dir=self.root))
        self.addCleanup(shutil.rmtree, self.work, True)
        self.addCleanup(setattr, train, "extract_frame", train.extract_frame)


class CacheReuseTest(CacheFixture):
    def test_a_matching_cache_stops_the_extraction(self):
        results = _run(self.frame, self.work)
        self.assertTrue((self.work / "features.npz").is_file())
        counter = _CountingExtractor()
        train.extract_frame = counter
        again = _run(self.frame, self.work)
        self.assertEqual(counter.calls, 0)
        self.assertEqual(
            results["test"]["roc_auc"], again["test"]["roc_auc"],
            "a warm run must reproduce the cold run exactly",
        )

    def test_a_second_seed_reads_the_runs_cache_instead_of_its_own_empty_one(self):
        _run(self.frame, self.work / "primary", seed=42)
        before = (self.work / "primary" / "features.npz").read_bytes()
        counter = _CountingExtractor()
        train.extract_frame = counter
        _run(self.frame, self.work / "seed_7", seed=7, feature_cache_root=self.work / "primary")
        self.assertEqual(
            counter.calls, 0,
            "the seed directory has no cache of its own, so without a shared read "
            "root this is where a full corpus extraction happens",
        )
        # the seed still writes its own copy, because the split stored inside it
        # is the seed's and must not be mistaken for the primary run's
        self.assertTrue((self.work / "seed_7" / "features.npz").is_file())
        self.assertEqual(
            before,
            (self.work / "primary" / "features.npz").read_bytes(),
            "a variance pass must not rewrite the evidence the headline cites",
        )

    def test_without_a_shared_read_root_the_seed_would_extract(self):
        """The bug this fix removes, kept as a test so it cannot come back quietly."""
        _run(self.frame, self.work / "primary", seed=42)
        counter = _CountingExtractor()
        train.extract_frame = counter
        with self.assertRaises(AssertionError):
            _run(self.frame, self.work / "seed_13", seed=13)
        self.assertEqual(counter.calls, 1)


class CacheRefusalTest(CacheFixture):
    def test_changed_pixels_invalidate_the_cache(self):
        _run(self.frame, self.work)
        target = Path(str(self.frame["image_path"].iloc[0]))
        original = target.read_bytes()
        target.write_bytes(original + b"\x00\x01\x02\x03")
        self.addCleanup(target.write_bytes, original)
        import features

        matrix = np.zeros((len(self.frame), len(features.feature_names())))
        counter = _CountingExtractor(matrix=matrix)
        train.extract_frame = counter
        _run(self.frame, self.work)
        self.assertEqual(counter.calls, 1, "a content change must invalidate the cache")

    def test_a_reordered_frame_is_a_cache_miss_by_design(self):
        """The check is positional, so reordering rows costs a re-extraction.

        Recorded rather than fixed: making it order-independent needs a per-row
        identity, and sample ids are positional in synthetic mode - the same id
        can point at different pixels in two builds, which is exactly why the
        content digest sits in the key beside it.
        """
        _run(self.frame, self.work)
        shuffled = self.frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
        import features

        matrix = np.zeros((len(shuffled), len(features.feature_names())))
        counter = _CountingExtractor(matrix=matrix)
        train.extract_frame = counter
        _run(shuffled, self.work)
        self.assertEqual(counter.calls, 1)

    def test_a_cache_written_by_another_feature_version_is_refused(self):
        _run(self.frame, self.work)
        path = self.work / "features.npz"
        with np.load(path, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
        payload["feature_version"] = np.asarray("000000000000")
        np.savez_compressed(path, **payload)
        counter = _CountingExtractor()
        train.extract_frame = counter
        with self.assertRaises(AssertionError):
            _run(self.frame, self.work)
        self.assertEqual(counter.calls, 1)


if __name__ == "__main__":
    unittest.main()
