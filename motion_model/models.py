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


class TemporalConvVAE(nn.Module):
    def __init__(self, in_dim: int = 56, latent_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Conv1d(128, latent_dim, kernel_size=3, padding=1)
        self.logvar_head = nn.Conv1d(128, latent_dim, kernel_size=3, padding=1)
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, in_dim, kernel_size=3, padding=1),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x.transpose(1, 2))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        y = self.decoder(z)
        return y.transpose(1, 2)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y = self.decode(z)
        return y, mu.transpose(1, 2), logvar.transpose(1, 2), z.transpose(1, 2)
