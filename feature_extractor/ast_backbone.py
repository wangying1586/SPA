# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class ASTBackbone(nn.Module):
    """
    AST feature extractor for raw waveform input.
    Input waveform: [B, T], 16 kHz. Output feature: [B, hidden_dim].
    """

    def __init__(
        self,
        model_path: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        freeze_encoder: bool = False,
        target_frames: int = 1024,
        sample_rate: int = 16000,
    ):
        super().__init__()
        try:
            from transformers import ASTModel
        except Exception as e:
            raise ImportError("ASTBackbone requires transformers. Install with: pip install transformers") from e

        self.encoder = ASTModel.from_pretrained(model_path)
        self.final_feat_dim = int(self.encoder.config.hidden_size)
        self.target_frames = int(target_frames)
        self.sample_rate = int(sample_rate)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _waveform_to_fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        feats = []
        for wav in waveform:
            fb = torchaudio.compliance.kaldi.fbank(
                wav.detach().cpu().unsqueeze(0),
                htk_compat=True,
                sample_frequency=self.sample_rate,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10,
            )
            if fb.shape[0] < self.target_frames:
                fb = F.pad(fb, (0, 0, 0, self.target_frames - fb.shape[0]))
            else:
                fb = fb[: self.target_frames]
            fb = (fb - (-4.2677393)) / (4.5689974 * 2.0)
            feats.append(fb)

        return torch.stack(feats, dim=0).to(waveform.device)

    def forward(self, waveform: torch.Tensor, training: bool = False) -> torch.Tensor:
        x = self._waveform_to_fbank(waveform)
        out = self.encoder(input_values=x)
        return out.last_hidden_state[:, 0]
