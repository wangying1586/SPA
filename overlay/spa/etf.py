from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def simplex_etf(feature_dim: int, num_classes: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Construct a d x K simplex ETF with unit-norm columns."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if feature_dim < num_classes:
        raise ValueError(
            f"feature_dim ({feature_dim}) must be no smaller than num_classes ({num_classes})"
        )

    identity = torch.eye(num_classes, dtype=dtype)
    centering = identity - torch.full(
        (num_classes, num_classes), 1.0 / num_classes, dtype=dtype
    )
    if feature_dim > num_classes:
        centering = torch.cat(
            [centering, torch.zeros(feature_dim - num_classes, num_classes, dtype=dtype)],
            dim=0,
        )
    return F.normalize(centering, p=2, dim=0)


def etf_gram(num_classes: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    identity = torch.eye(num_classes, dtype=dtype)
    ones = torch.ones(num_classes, num_classes, dtype=dtype)
    return (num_classes / (num_classes - 1.0)) * (
        identity - ones / num_classes
    )
