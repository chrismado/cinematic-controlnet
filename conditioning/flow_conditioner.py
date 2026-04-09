"""
Flow Conditioner

Takes dense optical flow matrices from the NeuralContinuumSolver
and injects them as conditioning signals into a diffusion model.

Compatible with HunyuanVideo 1.5 and Wan2.2 as diffusion backends.
Flow conditioning follows the ControlNet-style zero-conv injection
pattern but operates on physics-derived flow rather than 2D edge maps.

Reference: RealWonder (Liu et al., March 2026), arxiv 2603.05449
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroConv(nn.Module):
    """Zero-initialized convolution for residual conditioning injection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FlowConditioner(nn.Module):
    """
    Converts optical flow + coarse RGB into conditioning tensors
    compatible with diffusion model cross-attention or additive injection.

    Supports two backends:
      - "hunyuan": HunyuanVideo 1.5 (default hidden_dim=1280)
      - "wan2": Wan2.2 (default hidden_dim=1024)
    """

    def __init__(
        self,
        flow_channels: int = 2,
        rgb_channels: int = 3,
        hidden_dim: int = 256,
        conditioning_dim: int = 1280,
        backend: str = "hunyuan",
    ):
        super().__init__()
        self.backend = backend

        if backend == "wan2":
            conditioning_dim = conditioning_dim if conditioning_dim != 1280 else 1024

        in_channels = flow_channels + rgb_channels  # 5

        # Encoder: fuse flow + RGB into feature maps
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.SiLU(),
        )

        # Project to diffusion model's conditioning dimension
        self.proj = nn.Conv2d(hidden_dim, conditioning_dim, 1)

        # Zero-conv for residual injection (ControlNet pattern)
        self.zero_conv = ZeroConv(conditioning_dim, conditioning_dim)

    def forward(
        self,
        optical_flow: torch.Tensor,
        coarse_rgb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            optical_flow: Dense flow from physics solver [B, H, W, 2]
            coarse_rgb: Structural RGB preview [B, H, W, 3]
        Returns:
            conditioning: Tensor for diffusion injection [B, C, h, w]
                where C = conditioning_dim and h, w are spatially downsampled
        """
        # Reshape from [B, H, W, C] to [B, C, H, W] for convolutions
        flow = optical_flow.permute(0, 3, 1, 2)
        rgb = coarse_rgb.permute(0, 3, 1, 2)

        # Concatenate flow + RGB along channel dim [B, 5, H, W]
        fused = torch.cat([flow, rgb], dim=1)

        # Encode to feature maps [B, hidden_dim, H/4, W/4]
        features = self.encoder(fused)

        # Project to diffusion conditioning dim [B, conditioning_dim, H/4, W/4]
        projected = self.proj(features)

        # Zero-conv gated injection
        conditioning = self.zero_conv(projected)

        return conditioning

    def inject_into_diffusion(
        self,
        conditioning: torch.Tensor,
        diffusion_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Additive injection of conditioning signal into diffusion backbone features.

        Args:
            conditioning: Output of forward() [B, C, h, w]
            diffusion_features: Intermediate features from diffusion model [B, C, h', w']
        Returns:
            Modified diffusion features with physics conditioning [B, C, h', w']
        """
        # Spatially align if dimensions differ
        if conditioning.shape[2:] != diffusion_features.shape[2:]:
            conditioning = F.interpolate(
                conditioning,
                size=diffusion_features.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        return diffusion_features + conditioning
