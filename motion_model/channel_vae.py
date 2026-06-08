#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelTemporalVAE(nn.Module):
    def __init__(self, input_dim: int, target_len: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = int(input_dim)
        self.target_len = int(target_len)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)

        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, hidden_dim * target_len)
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, self.input_dim, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x.transpose(1, 2))
        pooled = h.mean(dim=2)
        return self.mu(pooled), self.logvar(pooled)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor):
        h = self.dec_fc(z).view(z.shape[0], self.hidden_dim, self.target_len)
        y = self.decoder(h)
        return y.transpose(1, 2)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
