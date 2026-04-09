"""
HunyuanVideo 1.5 ControlNet Adapter

Zero-convolution initialized adapter for injecting physics-based
conditioning into the HunyuanVideo 1.5 diffusion backbone.

Follows the ControlNet pattern: zero-conv ensures the adapter has
no effect at initialization, then gradually learns conditioning.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroConv2d(nn.Module):
    """2D convolution initialized to zero for residual injection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ZeroLinear(nn.Module):
    """Linear layer initialized to zero for residual injection."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class AdapterBlock(nn.Module):
    """Single adapter block with residual zero-conv injection."""

    def __init__(self, channels: int, cond_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels + cond_channels, channels, 3, padding=1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.zero_conv = ZeroConv2d(channels, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] diffusion backbone features
            cond: [B, C_cond, H, W] conditioning features
        """
        h = self.norm(x)
        # Align spatial dimensions
        if cond.shape[2:] != h.shape[2:]:
            cond = F.interpolate(cond, size=h.shape[2:], mode="bilinear", align_corners=False)
        h = torch.cat([h, cond], dim=1)
        h = self.act(self.conv1(h))
        h = self.conv2(h)
        return x + self.zero_conv(h)


class HunyuanVideoAdapter(nn.Module):
    """
    ControlNet-style adapter for HunyuanVideo 1.5.

    Injects physics conditioning into the diffusion backbone via
    zero-initialized convolutions at multiple resolution levels.
    """

    def __init__(
        self,
        model_channels: int = 1280,
        cond_channels: int = 1280,
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

        # Adapter blocks at different scales
        self.blocks = nn.ModuleList([AdapterBlock(model_channels, model_channels) for _ in range(num_blocks)])

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
            x: [B, C, H, W] noisy latent from diffusion backbone
            controlnet_cond: [B, C_cond, h, w] physics conditioning
            timestep: [B, timestep_dim] diffusion timestep embedding
        Returns:
            residual: [B, C, H, W] conditioning residual to add to backbone
        """
        # Timestep conditioning
        t_emb = self.time_embed(timestep)  # [B, C]
        t_emb = t_emb[:, :, None, None]  # [B, C, 1, 1]

        # Project conditioning
        cond = self.cond_input(controlnet_cond)

        # Add timestep to input
        h = x + t_emb

        # Process through adapter blocks
        for block in self.blocks:
            h = block(h, cond)

        return self.output_zero_conv(h)

    @classmethod
    def load_base_model(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> "HunyuanVideoAdapter":
        """
        Load adapter weights from a checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint
            device: Target device
            **kwargs: Override constructor arguments
        Returns:
            Loaded HunyuanVideoAdapter
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(str(path), map_location=device, weights_only=True)

        if "config" in checkpoint:
            config = checkpoint["config"]
            config.update(kwargs)
            model = cls(**config)
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model = cls(**kwargs)
            model.load_state_dict(checkpoint)

        return model.to(device).eval()
