#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalConvAE(nn.Module):
    def __init__(self, in_dim: int = 56, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, latent_dim, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, in_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor):
        # x [B,T,56]
        x1 = x.transpose(1, 2)
        z = self.encoder(x1)
        y = self.decoder(z)
        return y.transpose(1, 2), z.transpose(1, 2)
