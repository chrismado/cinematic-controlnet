"""
Wan2.2 ControlNet Adapter

Zero-convolution initialized adapter for injecting physics-based
conditioning into the Wan2.2 diffusion backbone.

Similar architecture to HunyuanVideoAdapter but configured for
Wan2.2's channel dimensions (1024 vs 1280).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroConv2d(nn.Module):
    """2D convolution initialized to zero."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Wan2AdapterBlock(nn.Module):
    """Adapter block tuned for Wan2.2 architecture."""

    def __init__(self, channels: int, cond_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels + cond_channels, channels, 3, padding=1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.zero_conv = ZeroConv2d(channels, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if cond.shape[2:] != h.shape[2:]:
            cond = F.interpolate(cond, size=h.shape[2:], mode="bilinear", align_corners=False)
        h = torch.cat([h, cond], dim=1)
        h = self.act(self.conv1(h))
        h = self.conv2(h)
        return x + self.zero_conv(h)


class Wan2Adapter(nn.Module):
    """
    ControlNet-style adapter for Wan2.2 diffusion model.

    Configured for Wan2.2's default channel dimension of 1024.
    Uses zero-initialized convolutions for safe residual injection.
    """

    def __init__(
        self,
        model_channels: int = 1024,
        cond_channels: int = 1024,
        num_blocks: int = 4,
        timestep_dim: int = 256,
    ):
        super().__init__()
        self.model_channels = model_channels

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(timestep_dim, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )

        # Input conditioning projection
        self.cond_input = nn.Sequential(
            nn.Conv2d(cond_channels, model_channels, 1),
            nn.SiLU(),
        )

        # Adapter blocks
        self.blocks = nn.ModuleList([
            Wan2AdapterBlock(model_channels, model_channels)
            for _ in range(num_blocks)
        ])

        # Final zero-conv output
        self.output_zero_conv = ZeroConv2d(model_channels, model_channels)

    def forward(
        self,
        x: torch.Tensor,
        controlnet_cond: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] noisy latent from Wan2.2 backbone
            controlnet_cond: [B, C_cond, h, w] physics conditioning
            timestep: [B, timestep_dim] diffusion timestep embedding
        Returns:
            residual: [B, C, H, W] conditioning residual
        """
        t_emb = self.time_embed(timestep)[:, :, None, None]
        cond = self.cond_input(controlnet_cond)

        h = x + t_emb
        for block in self.blocks:
            h = block(h, cond)

        return self.output_zero_conv(h)
