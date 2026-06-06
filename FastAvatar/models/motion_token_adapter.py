import torch
import torch.nn as nn


class MotionTokenAdapter(nn.Module):
    """Project a compact motion/channel latent vector into FastAvatar transformer dim."""

    def __init__(self, input_dim: int = 96, hidden_dim: int = 1024, output_dim: int = 1024, activation: str = "silu"):
        super().__init__()
        if activation == "relu":
            act = nn.ReLU()
        else:
            act = nn.SiLU()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            act,
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, motion_token_input: torch.Tensor) -> torch.Tensor:
        if motion_token_input.ndim != 2:
            raise ValueError(f"motion_token_input must be [B, D], got shape {list(motion_token_input.shape)}")
        if motion_token_input.shape[-1] != self.input_dim:
            raise ValueError(
                f"motion_token_input last dim must be {self.input_dim}, got {motion_token_input.shape[-1]}"
            )
        return self.net(motion_token_input)
