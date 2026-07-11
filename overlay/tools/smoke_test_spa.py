from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from spa import SPAHead, SPALoss


def main() -> None:
    torch.manual_seed(7)
    head = SPAHead(
        feature_dim=32,
        num_classes=4,
        momentum=0.99,
        temperature=2.0,
    )
    criterion = SPALoss(
        gamma=0.5,
        alignment_weight=0.5,
        distillation_temperature=2.0,
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    features = torch.randn(16, 12, 32)
    labels = torch.tensor([0, 1, 2, 3] * 4)
    output = head(features)
    terms = criterion(output.logits_s, output.logits_g, labels)
    optimizer.zero_grad(set_to_none=True)
    terms.total.backward()
    optimizer.step()
    updated = head.update_geometry(output.normalized_features, labels)

    assert output.logits_s.shape == (16, 4)
    assert output.logits_g.shape == (16, 4)
    assert output.fused_probabilities.shape == (16, 4)
    assert torch.isfinite(terms.total)
    assert updated
    orthogonality = head.rotation.T @ head.rotation
    assert torch.allclose(orthogonality, torch.eye(32), atol=1e-4, rtol=1e-4)
    print("SPA smoke test passed")


if __name__ == "__main__":
    main()
