"""
Latent RGB Renderer

Decodes latent representations combined with optical flow into
coarse structural RGB frames using transposed convolutions.
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentRGBRenderer(nn.Module):
    """
    Renders RGB frames from latent features and optical flow.

    Uses ConvTranspose2d layers with GELU activation for progressive
    upsampling from latent space to pixel space.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        flow_channels: int = 2,
        hidden_dim: int = 256,
        output_h: int = 480,
        output_w: int = 832,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_h = output_h
        self.output_w = output_w

        # Initial spatial size before upsampling
        self.init_h = output_h // 16
        self.init_w = output_w // 16

        # Project latent to initial spatial feature map
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * self.init_h * self.init_w),
            nn.GELU(),
        )

        # Flow encoder
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(flow_channels, hidden_dim // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, 3, stride=2, padding=1),
            nn.GELU(),
        )

        in_ch = hidden_dim + hidden_dim // 4

        # Progressive upsampling decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_ch, hidden_dim, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 8, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        latent: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            latent: [B, latent_dim] latent representation
            flow: [B, 2, H, W] optical flow field
        Returns:
            rgb: [B, 3, output_H, output_W] rendered RGB frame
        """
        B = latent.shape[0]

        # Project latent to spatial feature map
        lat_flat = self.latent_proj(latent)
        lat_map = lat_flat.view(B, -1, self.init_h, self.init_w)

        # Encode flow and downsample to match latent spatial size
        flow_feat = self.flow_encoder(flow)
        flow_feat = F.interpolate(
            flow_feat,
            size=(self.init_h, self.init_w),
            mode="bilinear",
            align_corners=False,
        )

        # Fuse latent + flow features
        fused = torch.cat([lat_map, flow_feat], dim=1)

        # Decode to RGB
        rgb = self.decoder(fused)

        # Ensure exact output size
        if rgb.shape[2:] != (self.output_h, self.output_w):
            rgb = F.interpolate(
                rgb,
                size=(self.output_h, self.output_w),
                mode="bilinear",
                align_corners=False,
            )

        return rgb

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "LatentRGBRenderer":
        """
        Load a pretrained LatentRGBRenderer from a checkpoint file.

        Args:
            checkpoint_path: Path to .pt or .safetensors checkpoint
            device: Device to load the model onto
            **kwargs: Override constructor arguments
        Returns:
            Loaded LatentRGBRenderer instance
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(str(path), map_location=device, weights_only=True)

        # Support both raw state_dict and wrapped checkpoint formats
        if "config" in checkpoint:
            config = checkpoint["config"]
            config.update(kwargs)
            model = cls(**config)
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model = cls(**kwargs)
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()
        return model
