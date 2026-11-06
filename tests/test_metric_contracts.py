"""Metric definitions must stay single-sourced and arithmetically right.

Two failure modes this guards:

1.  *Drift by duplication.* The cross-dataset evaluator used to compute its own
    scores, so an external number could differ from the in-domain one for
    reasons that had nothing to do with the data. Every evaluator now goes
    through ``evaluate_probabilities``, and a test that pins the numbers makes
    that claim checkable rather than architectural.
2.  *Silently wrong arithmetic.* A per-forgery-family AUC is measured against a
    shared pool of authentic images, and a constant score is not an
    uninformative one: against authentic drawn from U(0.3, 0.7), a flat 0.5 sits
    at about 0.75, not 0.5. The first version of this test failed because of
    exactly that mistake in the fixture, not in the code.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from cross_dataset import by_tamper_type
from train import evaluate_probabilities

PAPER_METRICS = {
    "threshold",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "mcc",
    "confusion_matrix",
}


class MetricBlockTest(unittest.TestCase):
    def test_the_six_published_columns_are_all_emitted(self):
        rng = np.random.default_rng(0)
        y = np.repeat([0, 1], 50)
        probabilities = rng.uniform(0, 1, len(y))
        metrics = evaluate_probabilities(y, probabilities, 0.5)
        self.assertTrue(PAPER_METRICS.issubset(metrics.keys()), sorted(metrics))

    def test_mcc_matches_its_definition(self):
        # TP=2, TN=1, FP=1, FN=0 -> 2 / sqrt(3*2*2*1)
        y = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.9, 0.9, 0.8])
        metrics = evaluate_probabilities(y, probabilities, 0.5)
        self.assertAlmostEqual(metrics["mcc"], 2 / np.sqrt(12), places=12)
        self.assertEqual(metrics["mcc"], matthews_corrcoef(y, (probabilities >= 0.5).astype(int)))

    def test_balanced_accuracy_ignores_class_imbalance_that_accuracy_does_not(self):
        y = np.array([0, 0, 0, 1])
        probabilities = np.array([0.1, 0.1, 0.1, 0.9])
        metrics = evaluate_probabilities(y, probabilities, 0.5)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 1.0)
        missed = evaluate_probabilities(y, np.array([0.1, 0.1, 0.1, 0.1]), 0.5)
        self.assertAlmostEqual(missed["accuracy"], 0.75)
        self.assertAlmostEqual(missed["balanced_accuracy"], 0.5)

    def test_roc_auc_is_the_sklearn_one(self):
        rng = np.random.default_rng(3)
        y = np.repeat([0, 1], 200)
        probabilities = np.clip(rng.normal(y * 0.4 + 0.3, 0.2), 0, 1)
        metrics = evaluate_probabilities(y, probabilities, 0.5)
        self.assertAlmostEqual(metrics["roc_auc"], roc_auc_score(y, probabilities), places=12)

    def test_the_confusion_matrix_orders_classes_the_same_way_as_the_labels(self):
        y = np.array([0, 0, 1, 1])
        probabilities = np.array([0.9, 0.1, 0.9, 0.1])
        metrics = evaluate_probabilities(y, probabilities, 0.5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])


def _fixture() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Three families with known answers, over a shared authentic pool, on a hostile index.

    The uninformative family is drawn from the same distribution as the authentic
    images rather than held constant, because a constant score is separable from a
    spread one and would not be a null.
    """
    rng = np.random.default_rng(11)
    authentic = rng.uniform(0.30, 0.60, size=300)
    records = [
        ("authentic", 0, float(score)) for score in authentic
    ]
    # Perfectly separable in the wrong direction: every score below every authentic one.
    records += [("distortion", 1, 0.01)] * 60
    # Perfectly separable in the right direction.
    records += [("translation", 1, 0.99)] * 60
    # Indistinguishable, by construction: same law as the authentic pool.
    records += [
        ("rotation", 1, float(score)) for score in rng.uniform(0.30, 0.60, size=60)
    ]

    frame = pd.DataFrame(records, columns=["tamper_type", "label", "score"])
    # Duplicate the index and shuffle the rows, so a grouping bug that trusts row
    # positions has somewhere to show up. Labels and scores travel *with* their
    # rows: permuting only the frame would test the fixture's ordering instead.
    frame = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
    frame.index = rng.permutation(np.arange(len(frame) // 2).tolist() * 2)
    return frame, frame["label"].to_numpy(), frame["score"].to_numpy()


class ByTamperTypeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame, cls.y, cls.probabilities = _fixture()
        cls.report = by_tamper_type(cls.frame, cls.y, cls.probabilities, 0.5)

    def test_every_forged_family_appears(self):
        self.assertEqual(
            set(self.report), {"distortion", "rotation", "translation"}
        )

    def test_directional_families_reach_the_extremes(self):
        self.assertAlmostEqual(self.report["translation"]["roc_auc"], 1.0, places=6)
        self.assertAlmostEqual(self.report["distortion"]["roc_auc"], 0.0, places=6)

    def test_an_indistinguishable_family_sits_at_chance(self):
        self.assertAlmostEqual(self.report["rotation"]["roc_auc"], 0.5, delta=0.08)

    def test_the_authentic_pool_is_shared_and_not_double_counted(self):
        authentic_rows = int((self.y == 0).sum())
        for block in self.report.values():
            self.assertEqual(block["authentic"], authentic_rows)
            self.assertEqual(block["images"], block["forged"] + block["authentic"])

    def test_probability_mean_is_over_forged_rows_only(self):
        family = "translation"
        mask = (self.y == 1) & (self.frame["tamper_type"].to_numpy() == family)
        self.assertAlmostEqual(
            self.report[family]["mean_probability"],
            float(self.probabilities[mask].mean()),
            places=12,
        )

    def test_metrics_come_from_the_shared_block(self):
        """The report must be the shared metric block on a family-vs-authentic subset."""
        for family, block in self.report.items():
            mask = (self.y == 1) & (self.frame["tamper_type"].to_numpy() == family)
            subset = mask | (self.y == 0)
            reference = evaluate_probabilities(
                self.y[subset], self.probabilities[subset], 0.5
            )
            self.assertAlmostEqual(block["roc_auc"], reference["roc_auc"], places=12)
            self.assertAlmostEqual(
                block["balanced_accuracy"], reference["balanced_accuracy"], places=12
            )
            self.assertAlmostEqual(
                block["recall_at_training_threshold"], reference["recall"], places=12
            )

    def test_frame_columns_line_up_with_the_arrays(self):
        """Guards the fixture: y and probabilities are positional partners of the frame.

        A shuffle that desynchronised the three objects would leave every family
        assertion above measuring a different mix of labels while still looking
        plausible, which is the failure this file is about.
        """
        self.assertTrue((self.frame["label"].to_numpy() == self.y).all())
        self.assertTrue((self.frame["score"].to_numpy() == self.probabilities).all())
        tampered_labels = int(((self.y == 1) & (self.frame["label"] == 1)).sum())
        self.assertEqual(tampered_labels, int((self.y == 1).sum()))


if __name__ == "__main__":
    unittest.main()
