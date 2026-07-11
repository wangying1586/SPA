from __future__ import annotations

import unittest

import numpy as np
import torch

from spa.alignment import orthogonal_procrustes_rotation
from spa.etf import etf_gram, simplex_etf
from spa.head import SPAHead
from spa.losses import SPALoss
from training.metrics import compute_classification_metrics


class TestSPA(unittest.TestCase):
    def test_simplex_etf_gram(self) -> None:
        matrix = simplex_etf(16, 4)
        self.assertTrue(
            torch.allclose(matrix.T @ matrix, etf_gram(4), atol=1e-6, rtol=1e-6)
        )

    def test_procrustes_recovers_rotation(self) -> None:
        torch.manual_seed(11)
        etf = simplex_etf(8, 4)
        q, _ = torch.linalg.qr(torch.randn(8, 8))
        prototypes = (q @ etf).T
        estimated = orthogonal_procrustes_rotation(prototypes, etf)
        aligned = estimated @ etf
        self.assertTrue(torch.allclose(aligned, q @ etf, atol=1e-5, rtol=1e-5))

    def test_head_and_loss(self) -> None:
        torch.manual_seed(3)
        head = SPAHead(32, 4)
        criterion = SPALoss()
        features = torch.randn(12, 5, 32)
        labels = torch.tensor([0, 1, 2, 3] * 3)
        output = head(features)
        terms = criterion(output.logits_s, output.logits_g, labels)
        self.assertEqual(output.logits_s.shape, (12, 4))
        self.assertEqual(output.logits_g.shape, (12, 4))
        self.assertTrue(torch.isfinite(terms.total))
        self.assertTrue(head.update_geometry(output.normalized_features, labels))

    def test_metrics(self) -> None:
        probabilities = np.array(
            [
                [0.9, 0.05, 0.03, 0.02],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.1, 0.1, 0.7],
            ]
        )
        labels = np.array([0, 1, 2, 3])
        metrics = compute_classification_metrics(probabilities, labels, 4)
        self.assertAlmostEqual(metrics["score"], 100.0)
        self.assertAlmostEqual(metrics["accuracy"], 100.0)


if __name__ == "__main__":
    unittest.main()
