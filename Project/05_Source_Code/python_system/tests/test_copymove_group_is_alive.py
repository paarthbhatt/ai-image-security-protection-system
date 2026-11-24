"""The copy-move group must stay alive.

This is a regression test for a defect that produced no error. The group used to
run ORB with ``cross_check=True`` against the same image, which can only return
the identity mapping, so every image got the same value for every statistic.
Feature selection drops a constant column silently, the pipeline reported "143
features" either way, and the run that fixed it changed the in-domain AUC by
-0.004 and the forensic-only AUC by +0.037. Zero variance is not a failure; it is
worse than a failure, because nothing fails.

So the invariant is not "the columns exist" - they existed while dead - but
"planted copy-move evidence has to show up in them".
"""

from __future__ import annotations

import unittest

import numpy as np

import features


def _textured(height: int = 256, width: int = 256, seed: int = 0) -> np.ndarray:
    """Natural-ish content: copy-move matching needs texture, flat images prove nothing."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(height, width), dtype=np.uint8).astype(np.float32)
    smooth = np.clip(
        base * 0.35
        + 0.35 * np.roll(base, 1, axis=0)
        + 0.3 * np.add.outer(np.linspace(0, 255, height), np.zeros(width)),
        0,
        255,
    ).astype(np.uint8)
    return np.stack([smooth, np.roll(smooth, 5, axis=1), np.roll(smooth, 9, axis=0)], axis=2)


def _with_copy_move(seed: int = 0) -> np.ndarray:
    """A pixel-exact duplicate at a translation of (144, 80).

    Both numbers are multiples of ``CM_STRIDE``, and that is not a detail: the
    matcher compares 16x16 windows sampled every 8 pixels, so a paste one pixel
    off the grid shares content but not windows, and the descriptors stop
    matching. A real forgery survives requantisation because the statistic is
    low-frequency DCT; a fixture only needs to survive being honest about where
    the blocks line up.
    """
    image = _textured(seed=seed).copy()
    patch = image[16:96, 16:96].copy()
    image[160:240, 96:176] = patch
    return image


def _column(name: str, matrix: np.ndarray) -> np.ndarray:
    return matrix[:, features.feature_names().index(name)]


class CopyMoveGroupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        names = features.feature_names()
        cls.columns = [f"cm_{stat}" for stat in features.CM_STATISTICS]
        for column in cls.columns:
            if column not in names:
                raise AssertionError(f"{column} is not a registered feature name")
        clean = [_textured(seed=seed) for seed in range(6)]
        moved = [_with_copy_move(seed=seed) for seed in range(6)]
        cls.clean = np.vstack([features.extract_features(image) for image in clean])
        cls.moved = np.vstack([features.extract_features(image) for image in moved])

    def test_the_group_is_registered(self):
        self.assertEqual(len(self.columns), len(features.CM_STATISTICS))
        self.assertGreaterEqual(len(self.columns), 6)

    def test_statistics_that_track_content_are_not_constant(self):
        """The exact shape of the dead group: every statistic, one repeated value.

        Only the four statistics that count image content are required to vary.
        ``cm_cluster_groups`` is 2 for all six fixture images and that is correct
        - the paste creates one dominant offset plus a spurious bucket - so
        asserting variety there would test the fixture, not the detector.
        """
        for column in (
            "cm_blocks",
            "cm_matched_pairs",
            "cm_cluster_max",
            "cm_offset_entropy",
        ):
            values = _column(column, self.moved)
            self.assertGreater(
                len(np.unique(values)),
                1,
                f"{column} returns the same value for every copy-move image: {values}",
            )

    def test_the_group_as_a_whole_is_not_silent(self):
        """The defect was total: eight columns, eight constants, no error raised."""
        constant = [
            column
            for column in self.columns
            if len(np.unique(_column(column, self.moved))) == 1
            and len(np.unique(_column(column, self.clean))) == 1
        ]
        self.assertLess(
            len(constant),
            len(self.columns),
            f"every copy-move statistic is constant: {constant}",
        )

    def test_planted_copy_move_produces_matching_blocks(self):
        blocks = _column("cm_blocks", self.moved)
        self.assertGreater(
            float(np.median(blocks)),
            0.0,
            f"a 64x64 exact duplicate produced no matched blocks (values {blocks})",
        )

    def test_matching_finds_far_more_in_a_duplication_than_in_texture(self):
        """Not a claim about real forgeries: a claim that the detector answers to one."""
        moved = _column("cm_matched_pairs", self.moved)
        clean = _column("cm_matched_pairs", self.clean)
        self.assertGreater(
            float(np.median(moved)),
            float(np.median(clean)),
            f"planted duplications did not beat clean texture ({moved} vs {clean})",
        )

    def test_a_shifted_duplicate_collapses_onto_one_offset(self):
        """Measured on this fixture: cluster_max ~12, groups ~2, entropy ~1.0 bit.

        The clustering is the difference between "some blocks look alike" and
        "a region was moved by a fixed vector", so the assertions are about
        concentration, never about raw match counts.
        """
        cluster_max = _column("cm_cluster_max", self.moved)
        groups = _column("cm_cluster_groups", self.moved)
        ratio = _column("cm_cluster_ratio", self.moved)
        entropy = _column("cm_offset_entropy", self.moved)
        self.assertGreaterEqual(float(np.median(cluster_max)), 5.0, cluster_max)
        self.assertLessEqual(float(np.median(groups)), 4.0, groups)
        self.assertGreaterEqual(float(np.median(ratio)), 0.3, ratio)
        self.assertLess(float(np.median(entropy)), 2.0, entropy)

    def test_clean_texture_produces_no_matches_at_all(self):
        """Without this, "it found pairs" would only mean "it finds pairs"."""
        self.assertEqual(float(np.median(_column("cm_matched_pairs", self.clean))), 0.0)

    def test_values_are_finite(self):
        for matrix in (self.clean, self.moved):
            for column in self.columns:
                values = _column(column, matrix)
                self.assertTrue(np.isfinite(values).all(), f"{column} produced a non-finite value")


if __name__ == "__main__":
    unittest.main()
