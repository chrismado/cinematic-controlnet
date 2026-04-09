"""
Neural Continuum Mechanics Solver

Replaces offline Blender physics engine with a real-time learned
approximator operating in the latent space.

Blender problem: 200-400ms rendering latency per frame breaks the
13.2 FPS real-time claim from RealWonder (Liu et al., March 2026).

This solver approximates:
  - Rigid body dynamics
  - Fluid dynamics
  - Cloth simulation
  - Granular materials

Target inference time: ~8ms per frame (vs ~300ms Blender)
This enables the live interactive director tools that Higgsfield and Decart need.

Reference: RealWonder (Liu, Chen, Li, Wang, Yu, Wu — Stanford/USC), arxiv 2603.05449
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralContinuumSolver(nn.Module):
    """
    Lightweight learned physics approximator.
    Operates in the diffusion model's latent space.
    Outputs dense optical flow matrices for action conditioning.
    """
    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 256,
        output_h: int = 60,
        output_w: int = 104,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_h = output_h
        self.output_w = output_w

        # Force encoder: maps 3D force vector → latent conditioning
        self.force_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),  # 3D force + 3D torque
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Dynamics predictor: predicts next latent state
        self.dynamics = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim, nhead=8, batch_first=True
            ),
            num_layers=4,
        )

        # Spatial projection: map transformer output to H*W spatial tokens
        self.spatial_proj = nn.Linear(latent_dim, output_h * output_w)

        # Flow decoder: latent → dense optical flow matrices
        self.flow_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),  # (u, v) flow components
        )

        # RGB decoder: latent → coarse structural RGB preview
        self.rgb_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # R, G, B
        )

    def forward(
        self,
        latent_state: torch.Tensor,
        force_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_state: Current scene representation [B, seq_len, latent_dim]
            force_vector: Applied 3D force + torque [B, 6]
        Returns:
            optical_flow: Dense flow matrices [B, H, W, 2]
            coarse_rgb: Structural RGB preview [B, H, W, 3]
        """
        B = latent_state.shape[0]
        H, W = self.output_h, self.output_w

        # Encode force into latent conditioning [B, latent_dim]
        force_cond = self.force_encoder(force_vector)

        # Broadcast force conditioning across all spatial tokens and add
        # [B, 1, latent_dim] + [B, seq_len, latent_dim] → [B, seq_len, latent_dim]
        conditioned = latent_state + force_cond.unsqueeze(1)

        # Predict dynamics via transformer [B, seq_len, latent_dim]
        dynamics_out = self.dynamics(conditioned)

        # Project to spatial grid: [B, seq_len, latent_dim] → [B, H*W, latent_dim]
        # Use spatial_proj to get attention weights over H*W positions,
        # then aggregate via einsum
        # spatial_weights: [B, seq_len, H*W]
        spatial_weights = self.spatial_proj(dynamics_out)
        spatial_weights = F.softmax(spatial_weights, dim=1)
        # Weighted combination: [B, H*W, latent_dim]
        spatial_tokens = torch.einsum(
            "bsn,bsd->bnd", spatial_weights, dynamics_out
        )

        # Decode optical flow [B, H*W, 2] → [B, H, W, 2]
        optical_flow = self.flow_decoder(spatial_tokens).view(B, H, W, 2)

        # Decode coarse RGB [B, H*W, 3] → [B, H, W, 3]
        coarse_rgb = self.rgb_decoder(spatial_tokens).view(B, H, W, 3)

        return optical_flow, coarse_rgb
