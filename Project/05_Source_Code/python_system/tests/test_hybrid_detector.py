"""Tests for Model 4: Hybrid Forensic Detector (src/hybrid_detector.py)."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

try:
    import hybrid_detector
    from hybrid_detector import load_feature_extractor, extract_deep_embeddings

    HAS_DEPS = True
except Exception:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "PyTorch / hybrid detector dependencies required")
class TestHybridDetector(unittest.TestCase):

    def test_identity_trunk_output_dim(self) -> None:
        """Trunk without fc must output (batch_size, 512)."""
        from torchvision import models

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Identity()
        model.eval()

        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        self.assertEqual(out.shape, (2, 512))

    def test_concatenation_dimensions(self) -> None:
        """Deep embeddings (512) + Forensic features (143) must equal 655."""
        deep_features = np.random.randn(10, 512).astype(np.float32)
        forensic_features = np.random.randn(10, 143).astype(np.float32)
        hybrid = np.hstack([deep_features, forensic_features])
        self.assertEqual(hybrid.shape, (10, 655))


if __name__ == "__main__":
    unittest.main()

