# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


class BEATsBackbone(nn.Module):
    """
    BEATs feature extractor for raw waveform input.

    It expects Microsoft's BEATs.py from one of:
      - --beats_repo path
      - ./feature_extractor/BEATs.py
      - ./third_party/unilm/beats/BEATs.py
      - ./BEATs/BEATs.py
    """

    def __init__(self, ckpt_path: str, repo_path: str = "", freeze_encoder: bool = False):
        super().__init__()
        self._add_beats_paths(repo_path)

        try:
            from BEATs import BEATs, BEATsConfig
        except Exception as e:
            raise ImportError(
                "Cannot import BEATs.py. Run setup_beats_ast_backbones.sh first, "
                "or set BEATS_REPO to a folder containing BEATs.py."
            ) from e

        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"BEATs checkpoint not found: {ckpt}. "
                "Run setup_beats_ast_backbones.sh or set BEATS_CKPT correctly."
            )

        state = torch.load(str(ckpt), map_location="cpu")
        if "cfg" not in state or "model" not in state:
            raise KeyError(f"{ckpt} does not look like an official BEATs checkpoint. Expected keys: cfg and model.")

        cfg = BEATsConfig(state["cfg"])
        self.encoder = BEATs(cfg)
        self.encoder.load_state_dict(state["model"], strict=False)
        self.final_feat_dim = int(cfg.encoder_embed_dim)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    @staticmethod
    def _add_beats_paths(repo_path: str = ""):
        candidates = []
        if repo_path:
            candidates.append(Path(repo_path))
        candidates += [
            Path("./feature_extractor"),
            Path("./third_party/unilm/beats"),
            Path("./unilm/beats"),
            Path("./BEATs"),
        ]
        for p in candidates:
            p = p.resolve()
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))

    def forward(self, waveform: torch.Tensor, training: bool = False) -> torch.Tensor:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        padding_mask = torch.zeros(waveform.shape, dtype=torch.bool, device=waveform.device)
        out = self.encoder.extract_features(waveform, padding_mask=padding_mask)
        feat_seq = out[0] if isinstance(out, tuple) else out
        feat = feat_seq.mean(dim=1) if feat_seq.dim() == 3 else feat_seq
        return feat
