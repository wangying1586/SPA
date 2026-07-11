from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("backbone."):
            key = key[9:]
        cleaned[key] = value
    return cleaned


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    backbone: nn.Module,
    projector: nn.Module,
    spa_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    best_metrics: dict[str, Any],
    args: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model": backbone.state_dict(),
        "projector": projector.state_dict(),
        "spa_head": spa_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_metrics": best_metrics,
        "args": args,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    backbone: nn.Module,
    projector: nn.Module,
    spa_head: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    backbone.load_state_dict(_strip_module_prefix(checkpoint["model"]), strict=strict)
    projector.load_state_dict(checkpoint["projector"], strict=strict)
    spa_head.load_state_dict(checkpoint["spa_head"], strict=strict)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def initialize_from_pafa_checkpoint(
    path: Path,
    *,
    backbone: nn.Module,
    projector: nn.Module,
) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    backbone.load_state_dict(_strip_module_prefix(checkpoint["model"]), strict=False)
    if checkpoint.get("projector") is not None:
        projector.load_state_dict(checkpoint["projector"], strict=False)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")
