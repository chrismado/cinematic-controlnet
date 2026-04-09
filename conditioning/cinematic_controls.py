"""
Cinematic Controls

Filmmaker-native control signals that go beyond standard ControlNet's
depth/canny maps. These controls encode camera lens specifications,
depth maps with artistic intent, and color grading LUTs.

These are transformed into diffusion conditioning tensors that can be
injected alongside the physics flow conditioning.

Reference: RealWonder (Liu et al., March 2026), arxiv 2603.05449
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class CinematicControls:
    """
    Filmmaker-native control parameters.

    Attributes:
        focal_length: Camera focal length in mm (e.g., 24.0 for wide, 85.0 for portrait)
        aperture: f-stop value controlling depth of field (e.g., 1.4 for shallow DoF)
        depth_map: Per-pixel depth tensor [H, W] or [B, H, W], values in meters
        lut_path: Path to a .cube or .npy LUT file for color grading
    """
    focal_length: float = 50.0
    aperture: float = 2.8
    depth_map: Optional[torch.Tensor] = field(default=None, repr=False)
    lut_path: Optional[str] = None


def load_lut(lut_path: str, device: torch.device) -> torch.Tensor:
    """
    Load a color grading LUT from .npy or .cube file.

    Returns:
        lut_tensor: [N, 3] RGB mapping table as a tensor
    """
    path = Path(lut_path)
    if path.suffix == ".npy":
        lut = np.load(str(path))
        return torch.from_numpy(lut).float().to(device)

    if path.suffix == ".cube":
        entries = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("TITLE"):
                    continue
                if line.startswith("LUT_SIZE") or line.startswith("DOMAIN"):
                    continue
                parts = line.split()
                if len(parts) == 3:
                    try:
                        entries.append([float(x) for x in parts])
                    except ValueError:
                        continue
        if not entries:
            raise ValueError(f"No valid LUT entries found in {lut_path}")
        return torch.tensor(entries, dtype=torch.float32, device=device)

    raise ValueError(f"Unsupported LUT format: {path.suffix}. Use .npy or .cube")


class CinematicControlEncoder(nn.Module):
    """Encodes CinematicControls into diffusion conditioning tensors."""

    def __init__(self, conditioning_dim: int = 1280):
        super().__init__()
        self.conditioning_dim = conditioning_dim

        # Camera params encoder: focal_length + aperture → conditioning
        self.camera_encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.SiLU(),
            nn.Linear(128, conditioning_dim),
        )

        # Depth map encoder: single-channel spatial → conditioning feature map
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, conditioning_dim, 3, stride=2, padding=1),
        )

        # LUT encoder: projects LUT statistics into conditioning vector
        self.lut_encoder = nn.Sequential(
            nn.Linear(6, 128),  # mean + std of RGB channels
            nn.SiLU(),
            nn.Linear(128, conditioning_dim),
        )

    def forward(self, controls: CinematicControls, device: torch.device) -> dict[str, torch.Tensor]:
        """
        Transform cinematic controls into conditioning tensors.

        Returns dict with keys:
            "camera": [1, conditioning_dim] camera parameter embedding
            "depth": [1, conditioning_dim, h, w] depth conditioning map (if depth_map provided)
            "lut": [1, conditioning_dim] color grading embedding (if lut_path provided)
        """
        result = {}

        # Camera parameters
        camera_params = torch.tensor(
            [[controls.focal_length / 100.0, controls.aperture / 22.0]],
            dtype=torch.float32,
            device=device,
        )
        result["camera"] = self.camera_encoder(camera_params)

        # Depth map
        if controls.depth_map is not None:
            depth = controls.depth_map.to(device).float()
            if depth.dim() == 2:
                depth = depth.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            elif depth.dim() == 3:
                depth = depth.unsqueeze(1)  # [B, 1, H, W]
            result["depth"] = self.depth_encoder(depth)

        # Color grading LUT
        if controls.lut_path is not None:
            lut = load_lut(controls.lut_path, device)
            # Summarize LUT as channel-wise mean and std
            lut_stats = torch.cat([lut.mean(dim=0), lut.std(dim=0)]).unsqueeze(0)
            result["lut"] = self.lut_encoder(lut_stats)

        return result


def apply_controls(
    controls: CinematicControls,
    encoder: CinematicControlEncoder,
    device: torch.device,
) -> torch.Tensor:
    """
    High-level API: transform CinematicControls into a single
    conditioning vector suitable for diffusion model injection.

    Returns:
        conditioning: [1, conditioning_dim] combined cinematic conditioning
    """
    tensors = encoder(controls, device)

    # Start with camera embedding (always present)
    combined = tensors["camera"]

    # Add LUT conditioning if available
    if "lut" in tensors:
        combined = combined + tensors["lut"]

    # If depth map present, global-pool it and add
    if "depth" in tensors:
        depth_pooled = tensors["depth"].mean(dim=[2, 3])  # [B, C]
        combined = combined + depth_pooled

    return combined
