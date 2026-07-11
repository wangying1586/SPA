from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .alignment import orthogonal_procrustes_rotation, update_prototypes
from .etf import simplex_etf


@dataclass(frozen=True)
class SPAOutput:
    logits_s: Tensor
    logits_g: Tensor
    fused_probabilities: Tensor
    normalized_features: Tensor


class SPAHead(nn.Module):
    """Spherical Procrustes Alignment classifier head."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        *,
        momentum: float = 0.99,
        temperature: float = 2.0,
        fusion_weight: float = 0.5,
        scale_init: float = 16.0,
        update_after_all_classes: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim < num_classes:
            raise ValueError("feature_dim must be no smaller than num_classes")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= fusion_weight <= 1.0:
            raise ValueError("fusion_weight must lie in [0, 1]")

        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.prototype_momentum = float(momentum)
        self.temperature = float(temperature)
        self.fusion_weight = float(fusion_weight)
        self.update_after_all_classes = bool(update_after_all_classes)

        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.spherical_weight = nn.Parameter(
            torch.empty(num_classes, feature_dim)
        )
        nn.init.normal_(self.spherical_weight, std=0.02)

        self.adapter = nn.Linear(feature_dim, feature_dim)
        nn.init.eye_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

        self.register_buffer("etf", simplex_etf(feature_dim, num_classes))
        self.register_buffer("prototypes", torch.zeros(num_classes, feature_dim))
        self.register_buffer("class_counts", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("rotation", torch.eye(feature_dim))

    @staticmethod
    def pool_features(features: Tensor) -> Tensor:
        if features.ndim == 2:
            return features
        if features.ndim == 3:
            return features.mean(dim=1)
        raise ValueError(
            f"SPA expects [B, D] or [B, T, D] features, received {tuple(features.shape)}"
        )

    def forward(self, features: Tensor) -> SPAOutput:
        pooled = self.pool_features(features)
        normalized = F.normalize(pooled, p=2, dim=1)
        normalized_weight = F.normalize(self.spherical_weight, p=2, dim=1)
        logits_s = self.scale.clamp_min(1e-6) * F.linear(
            normalized, normalized_weight
        )

        adapted = self.adapter(pooled)
        aligned_etf = self.rotation @ self.etf
        logits_g = adapted @ aligned_etf
        logits_g = logits_g / self.temperature

        prob_s = F.softmax(logits_s, dim=1)
        prob_g = F.softmax(logits_g, dim=1)
        fused = self.fusion_weight * prob_s + (
            1.0 - self.fusion_weight
        ) * prob_g

        return SPAOutput(
            logits_s=logits_s,
            logits_g=logits_g,
            fused_probabilities=fused,
            normalized_features=normalized,
        )

    @torch.no_grad()
    def update_geometry(self, normalized_features: Tensor, labels: Tensor) -> bool:
        update_prototypes(
            self.prototypes,
            self.class_counts,
            normalized_features.detach().float(),
            labels.detach(),
            self.prototype_momentum,
        )

        all_seen = bool(torch.all(self.class_counts > 0).item())
        if self.update_after_all_classes and not all_seen:
            return False

        usable = self.class_counts > 0
        if int(usable.sum().item()) < 2:
            return False

        if all_seen:
            prototypes = self.prototypes
            etf = self.etf
        else:
            prototypes = self.prototypes[usable]
            etf = self.etf[:, usable]

        if all_seen:
            rotation = orthogonal_procrustes_rotation(prototypes, etf)
        else:
            target = prototypes.transpose(0, 1)
            cross_covariance = target @ etf.transpose(0, 1)
            u, _, vh = torch.linalg.svd(cross_covariance, full_matrices=True)
            rotation = u @ vh

        self.rotation.copy_(rotation.to(self.rotation.dtype))
        return True
