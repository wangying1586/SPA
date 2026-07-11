from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def normalize_nonzero_rows(x: Tensor, eps: float = 1e-12) -> Tensor:
    norms = x.norm(p=2, dim=1, keepdim=True)
    normalized = x / norms.clamp_min(eps)
    return torch.where(norms > eps, normalized, x)


def orthogonal_procrustes_rotation(prototypes: Tensor, etf: Tensor) -> Tensor:
    """Solve min_R ||P^T - R M||_F subject to R^T R = I.

    Args:
        prototypes: K x d class prototype matrix.
        etf: d x K simplex ETF matrix.
    Returns:
        d x d orthogonal rotation matrix.
    """
    if prototypes.ndim != 2 or etf.ndim != 2:
        raise ValueError("prototypes and etf must be matrices")
    k, d = prototypes.shape
    if etf.shape != (d, k):
        raise ValueError(
            f"expected etf shape {(d, k)}, received {tuple(etf.shape)}"
        )

    target = prototypes.transpose(0, 1)
    cross_covariance = target @ etf.transpose(0, 1)
    u, _, vh = torch.linalg.svd(cross_covariance, full_matrices=True)
    rotation = u @ vh

    # Numerical roundoff can be larger in reduced precision. The update is
    # gradient-free, so keeping it in float32 is inexpensive and more stable.
    return rotation.to(dtype=prototypes.dtype)


@torch.no_grad()
def update_prototypes(
    prototypes: Tensor,
    class_counts: Tensor,
    features: Tensor,
    labels: Tensor,
    momentum: float,
) -> None:
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must lie in [0, 1)")
    if features.ndim != 2:
        raise ValueError("features must have shape [batch, feature_dim]")
    labels = labels.long().view(-1)
    if features.size(0) != labels.numel():
        raise ValueError("features and labels have different batch sizes")

    for class_index in labels.unique(sorted=True).tolist():
        mask = labels == class_index
        batch_mean = F.normalize(features[mask].mean(dim=0), p=2, dim=0)
        if class_counts[class_index] == 0:
            prototypes[class_index].copy_(batch_mean)
        else:
            prototypes[class_index].mul_(momentum).add_(
                batch_mean, alpha=1.0 - momentum
            )
            prototypes[class_index].copy_(
                F.normalize(prototypes[class_index], p=2, dim=0)
            )
        class_counts[class_index].add_(int(mask.sum().item()))
