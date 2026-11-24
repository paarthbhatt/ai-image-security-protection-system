"""Invariants of the modified-CASIA protocol arm (``src/modified_casia.py``).

The arm exists to make our CASIA number comparable with the base paper's and to
take away the file-format cue. Each test here guards one way that could go wrong
without anything visibly failing:

*  converting an image must not move its pixels, or every mask misaligns;
*  carrying ``source_id`` forward, or a photograph and its splice can land in
   different folds and the leakage check stops meaning anything;
*  rebuilding must reproduce the corpus, or the arm cannot be cited later;
*  the narrow ``--tampered-only`` reading must really leave BMP originals alone.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from tests import fixtures

import modified_casia


def _manifest(output: Path) -> pd.DataFrame:
    # keep_default_na, not fillna: an empty mask_path is the manifest's own "no
    # mask" value, and filling a float column with "" is deprecated in pandas.
    return pd.read_csv(output / "manifest.csv", keep_default_na=False)


class QualityFactorDrawTest(unittest.TestCase):
    def test_stays_inside_the_declared_range(self):
        for index in range(200):
            quality = modified_casia.draw_quality_factor(f"stem{index}", 42, 70, 95)
            self.assertGreaterEqual(quality, 70)
            self.assertLessEqual(quality, 95)

    def test_is_independent_of_neighbouring_images(self):
        """A hash draw, so an unrelated image entering the build cannot renumber this one."""
        before = modified_casia.draw_quality_factor("Tp_S_keepme_00001", 42)
        for other in range(50):
            modified_casia.draw_quality_factor(f"Tp_S_other_{other:05d}", 42)
        self.assertEqual(before, modified_casia.draw_quality_factor("Tp_S_keepme_00001", 42))

    def test_seed_changes_the_assignment(self):
        stems = [f"stem_{index}" for index in range(60)]
        first = [modified_casia.draw_quality_factor(stem, 42) for stem in stems]
        second = [modified_casia.draw_quality_factor(stem, 7) for stem in stems]
        self.assertNotEqual(first, second)

    def test_rejects_an_empty_range(self):
        with self.assertRaises(ValueError):
            modified_casia.draw_quality_factor("anything", 42, 90, 50)


class BuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.source = cls.root / "casia2"
        cls.written = fixtures.build_fake_casia(cls.source, scenes=8)
        cls.output = cls.root / "casia2_mod"
        cls.summary = modified_casia.build(
            source_root=cls.source,
            output_root=cls.output,
            seed=42,
            target_per_class=cls.written["tampered"],
            qf_low=70,
            qf_high=95,
        )
        cls.frame = _manifest(cls.output)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_every_referenced_file_exists(self):
        for column in ("image_path", "mask_path"):
            for value in self.frame[column]:
                text = str(value)
                if not text:
                    continue
                self.assertTrue(
                    (self.output / text).is_file(), f"{column} {text} does not resolve"
                )

    def test_class_counts_are_balanced_at_the_tampered_size(self):
        tampered = int((self.frame["label"] == 1).sum())
        authentic = int((self.frame["label"] == 0).sum())
        self.assertEqual(tampered, self.written["tampered"])
        self.assertEqual(authentic, tampered)
        self.assertEqual(self.summary["rows"], tampered + authentic)
        # The source has more originals than that, so the run really thinned them
        # instead of quietly keeping every row and calling it balanced.
        self.assertGreater(self.written["authentic"], authentic)

    def test_only_uncompressed_images_were_converted(self):
        converted = self.frame[self.frame["converted"] == True]  # noqa: E712
        self.assertEqual(len(converted), int(self.summary["converted"]))
        for original in converted["original_path"]:
            self.assertTrue(modified_casia.is_uncompressed(original), original)
        for original in self.frame[self.frame["converted"] != True]["original_path"]:  # noqa: E712
            self.assertFalse(modified_casia.is_uncompressed(original), original)

    def test_conversion_keeps_pixel_dimensions(self):
        """Masks are indexed by shape: a resized image would silently break localisation."""
        for _, row in self.frame[self.frame["converted"] == True].iterrows():  # noqa: E712
            with Image.open(self.output / str(row["original_path"])) as source:
                expected = source.size
            with Image.open(self.output / str(row["image_path"])) as result:
                self.assertEqual(result.size, expected, str(row["image_path"]))

    def test_converted_files_are_jpegs(self):
        for _, row in self.frame[self.frame["converted"] == True].iterrows():  # noqa: E712
            path = self.output / str(row["image_path"])
            self.assertEqual(path.suffix.lower(), ".jpg")
            with Image.open(path) as handle:
                self.assertEqual(handle.format, "JPEG")

    def test_quality_factors_are_recorded_and_inside_the_range(self):
        converted = self.frame[self.frame["converted"] == True]  # noqa: E712
        values = converted["qf"].astype(int)
        self.assertTrue(values.between(70, 95).all())
        # A range this narrow over this many images must not collapse to one value.
        self.assertGreater(values.nunique(), 1)
        untouched = self.frame[self.frame["converted"] != True]  # noqa: E712
        self.assertTrue((untouched["qf"] == "").all())

    def test_labels_and_families_survive(self):
        self.assertEqual(
            self.frame.loc[self.frame["label"] == 1, "tamper_type"].nunique(), 2
        )
        self.assertTrue((self.frame.loc[self.frame["label"] == 0, "tamper_type"] == "authentic").all())

    def test_scene_linkage_is_carried_forward_not_re_derived(self):
        """A converted image sits in a new directory; its scene id must come from the source frame."""
        source = self.frame["source_id"].astype(str)
        self.assertTrue(source.str.startswith("casia2:scene:").all(), source.unique()[:5])
        # A splice and the photograph it was cut from share a scene, so the
        # linkage is only intact if converted rows kept the id they arrived with.
        converted = self.frame[self.frame["converted"] == True]  # noqa: E712
        for _, row in converted.iterrows():
            self.assertIn("ani", str(row["source_id"]))

    def test_grouped_split_keeps_one_scene_in_one_fold(self):
        from sources import load_real_forgery_dataset

        frame = load_real_forgery_dataset(name="casia2_mod", root=self.output, seed=42)
        folds = frame.groupby("source_id")["split"].nunique()
        self.assertTrue((folds == 1).all(), "a scene was split across folds")
        # With splices and their source photographs linked, a tampered row and an
        # authentic row must be able to share a scene; if none do, linkage broke
        # and the leakage check above is vacuous.
        shared = frame.groupby("source_id")["label"].nunique()
        self.assertGreater(int((shared == 2).sum()), 0, "no scene links a forgery to its original")

    def test_manifest_paths_are_relative(self):
        """A committed corpus description must not name one machine's home directory."""
        for column in ("image_path", "mask_path", "original_path"):
            for value in self.frame[column].astype(str):
                if value in ("", "nan"):
                    continue
                self.assertFalse(value.startswith("/"), value)
                self.assertNotIn(str(Path.home()), value)

    def test_rebuild_is_reproducible(self):
        second = self.root / "casia2_mod_again"
        again = modified_casia.build(
            source_root=self.source,
            output_root=second,
            seed=42,
            target_per_class=int(self.written["tampered"]),
            qf_low=70,
            qf_high=95,
        )
        self.assertEqual(again["manifest_sha256"], self.summary["manifest_sha256"])

    def test_a_different_seed_rebalances_differently(self):
        third = self.root / "casia2_mod_seed7"
        again = modified_casia.build(
            source_root=self.source,
            output_root=third,
            seed=7,
            target_per_class=int(self.written["tampered"]),
            qf_low=70,
            qf_high=95,
        )
        self.assertNotEqual(again["manifest_sha256"], self.summary["manifest_sha256"])

    def test_refuses_to_overwrite_silently(self):
        with self.assertRaises(SystemExit):
            modified_casia.build(
                source_root=self.source, output_root=self.output, seed=42, target_per_class=16
            )


class TamperedOnlyModeTest(unittest.TestCase):
    """The two readings of section 4.1.3, run over the same source tree.

    The paper's sentence is about tampered images but names "the uncompressed
    format images", which includes the originals. The fixture deliberately puts
    BMP files on the authentic side, because that is the disagreement: under the
    narrow reading "is BMP" stays a cue that only ever fires on authentic input.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.source = cls.root / "casia2"
        cls.written = fixtures.build_fake_casia(cls.source, scenes=8)
        cls.broad = cls.root / "broad"
        cls.narrow = cls.root / "narrow"
        common = {
            "source_root": cls.source,
            "seed": 42,
            "target_per_class": cls.written["tampered"],
        }
        modified_casia.build(output_root=cls.broad, **common)
        modified_casia.build(output_root=cls.narrow, tampered_only=True, **common)
        cls.broad_frame = _manifest(cls.broad)
        cls.narrow_frame = _manifest(cls.narrow)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_narrow_reading_converts_nothing_on_the_authentic_side(self):
        authentic = self.narrow_frame[self.narrow_frame["label"] == 0]
        self.assertFalse(bool((authentic["converted"] == True).any()))  # noqa: E712
        forged = self.narrow_frame[self.narrow_frame["label"] == 1]
        self.assertTrue(bool((forged["converted"] == True).any()))  # noqa: E712

    def test_broad_reading_also_converts_the_uncompressed_originals(self):
        authentic = self.broad_frame[self.broad_frame["label"] == 0]
        converted = authentic[authentic["converted"] == True]  # noqa: E712
        # Every converted authentic row must really have been uncompressed, and
        # with 8 of 24 originals as BMP one of them surviving a 16-row subsample
        # is not luck, it is the point of the mode.
        self.assertGreater(len(converted), 0)
        for original in converted["original_path"]:
            self.assertTrue(modified_casia.is_uncompressed(original), original)

    def test_the_two_readings_differ_in_row_count_only_by_conversions(self):
        self.assertEqual(len(self.broad_frame), len(self.narrow_frame))
        self.assertNotEqual(
            int((self.broad_frame["converted"] == True).sum()),  # noqa: E712
            int((self.narrow_frame["converted"] == True).sum()),  # noqa: E712
        )

    def test_both_readings_keep_the_same_labels(self):
        for frame in (self.broad_frame, self.narrow_frame):
            self.assertEqual(
                int((frame["label"] == 1).sum()), self.written["tampered"]
            )


if __name__ == "__main__":
    unittest.main()
