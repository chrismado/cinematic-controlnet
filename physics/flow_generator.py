"""
Optical Flow Generator

U-Net encoder-decoder with skip connections that generates dense optical
flow fields from physics state and previous frame information.

Output: [B, 2, H, W] flow field (horizontal and vertical displacement).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_to_rgb(flow: torch.Tensor, max_mag: float = -1.0) -> torch.Tensor:
    """
    Convert optical flow [B, 2, H, W] to RGB visualization [B, 3, H, W].

    Uses HSV encoding where hue = direction, saturation = 1, value = magnitude.
    """
    B, _, H, W = flow.shape
    u = flow[:, 0]  # [B, H, W]
    v = flow[:, 1]

    mag = torch.sqrt(u**2 + v**2)
    if max_mag <= 0:
        max_mag = mag.max().item() + 1e-8

    ang = torch.atan2(v, u)  # [-pi, pi]
    hue = (ang + torch.pi) / (2 * torch.pi)  # [0, 1]
    sat = torch.ones_like(hue)
    val = torch.clamp(mag / max_mag, 0.0, 1.0)

    # HSV to RGB conversion
    h6 = hue * 6.0
    i = h6.long() % 6
    f = h6 - h6.floor()
    p = val * (1 - sat)
    q = val * (1 - sat * f)
    t = val * (1 - sat * (1 - f))

    r = torch.zeros_like(hue)
    g = torch.zeros_like(hue)
    b = torch.zeros_like(hue)

    mask0 = i == 0
    mask1 = i == 1
    mask2 = i == 2
    mask3 = i == 3
    mask4 = i == 4
    mask5 = i == 5

    r[mask0] = val[mask0]
    g[mask0] = t[mask0]
    b[mask0] = p[mask0]
    r[mask1] = q[mask1]
    g[mask1] = val[mask1]
    b[mask1] = p[mask1]
    r[mask2] = p[mask2]
    g[mask2] = val[mask2]
    b[mask2] = t[mask2]
    r[mask3] = p[mask3]
    g[mask3] = q[mask3]
    b[mask3] = val[mask3]
    r[mask4] = t[mask4]
    g[mask4] = p[mask4]
    b[mask4] = val[mask4]
    r[mask5] = val[mask5]
    g[mask5] = p[mask5]
    b[mask5] = q[mask5]

    return torch.stack([r, g, b], dim=1)  # [B, 3, H, W]


class DownBlock(nn.Module):
    """Encoder block: Conv -> SiLU -> Conv -> SiLU -> MaxPool."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.SiLU(),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features  # pooled goes deeper, features for skip


class UpBlock(nn.Module):
    """Decoder block: Upsample -> Cat skip -> Conv -> SiLU -> Conv -> SiLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class OpticalFlowGenerator(nn.Module):
    """
    U-Net encoder-decoder with skip connections for optical flow generation.

    Takes physics state embedding and previous frame, produces dense
    optical flow [B, 2, H, W].
    """

    def __init__(
        self,
        state_dim: int = 512,
        prev_frame_channels: int = 3,
        base_channels: int = 64,
        output_h: int = 60,
        output_w: int = 104,
    ):
        super().__init__()
        self.output_h = output_h
        self.output_w = output_w

        # Project physics state to spatial feature map
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, base_channels * (output_h // 4) * (output_w // 4)),
            nn.SiLU(),
        )

        in_ch = prev_frame_channels + base_channels  # concat prev_frame + state map

        # Encoder
        self.down1 = DownBlock(in_ch, base_channels)
        self.down2 = DownBlock(base_channels, base_channels * 2)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.SiLU(),
        )

        # Decoder
        self.up3 = UpBlock(base_channels * 8, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels)

        # Final conv to 2-channel flow
        self.out_conv = nn.Conv2d(base_channels, 2, 1)

    def forward(
        self,
        physics_state: torch.Tensor,
        prev_frame: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            physics_state: [B, state_dim] physics solver output
            prev_frame: [B, 3, H, W] previous generated frame
        Returns:
            flow: [B, 2, H, W] dense optical flow
        """
        B = physics_state.shape[0]
        H, W = self.output_h, self.output_w

        # Project state to spatial map [B, base_ch, H/4, W/4]
        state_flat = self.state_proj(physics_state)
        ch = self.down1.conv[0].in_channels - 3  # base_channels
        state_map = state_flat.view(B, ch, H // 4, W // 4)
        # Upsample state map to match prev_frame resolution
        state_map = F.interpolate(state_map, size=(H, W), mode="bilinear", align_corners=False)

        # Resize prev_frame to target resolution if needed
        if prev_frame.shape[2:] != (H, W):
            prev_frame = F.interpolate(prev_frame, size=(H, W), mode="bilinear", align_corners=False)

        # Concatenate along channel dimension
        x = torch.cat([prev_frame, state_map], dim=1)

        # Encoder with skip connections
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        # Output flow
        flow = self.out_conv(x)
        return flow  # [B, 2, H, W]
