from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SPALossOutput:
    total: Tensor
    ce_s: Tensor
    ce_g: Tensor
    kd: Tensor


class SPALoss(nn.Module):
    def __init__(
        self,
        *,
        gamma: float = 0.5,
        alignment_weight: float = 0.5,
        distillation_temperature: float = 2.0,
        label_smoothing: float = 0.1,
        kd_direction: str = "geo_to_sphere",
    ) -> None:
        super().__init__()
        if gamma < 0 or alignment_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if distillation_temperature <= 0:
            raise ValueError("distillation_temperature must be positive")
        if kd_direction not in {"geo_to_sphere", "symmetric"}:
            raise ValueError("kd_direction must be geo_to_sphere or symmetric")

        self.gamma = float(gamma)
        self.alignment_weight = float(alignment_weight)
        self.distillation_temperature = float(distillation_temperature)
        self.label_smoothing = float(label_smoothing)
        self.kd_direction = kd_direction

    def _kl(self, student: Tensor, teacher: Tensor) -> Tensor:
        tau = self.distillation_temperature
        return F.kl_div(
            F.log_softmax(student / tau, dim=1),
            F.softmax(teacher.detach() / tau, dim=1),
            reduction="batchmean",
        ) * (tau * tau)

    def forward(self, logits_s: Tensor, logits_g: Tensor, labels: Tensor) -> SPALossOutput:
        ce_s = F.cross_entropy(
            logits_s, labels, label_smoothing=self.label_smoothing
        )
        ce_g = F.cross_entropy(
            logits_g, labels, label_smoothing=self.label_smoothing
        )

        if self.kd_direction == "geo_to_sphere":
            kd = self._kl(logits_s, logits_g)
        else:
            kd = 0.5 * (
                self._kl(logits_s, logits_g) + self._kl(logits_g, logits_s)
            )

        total = self.gamma * (ce_s + ce_g) + self.alignment_weight * kd
        return SPALossOutput(total=total, ce_s=ce_s, ce_g=ce_g, kd=kd)
