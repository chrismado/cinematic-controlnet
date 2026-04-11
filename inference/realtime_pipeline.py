"""
Real-Time Inference Pipeline

Chains the full generation loop:
  force_vector → NeuralContinuumSolver → FlowConditioner → diffusion → output frame

Target: 13.2 FPS at 480x832 resolution.

Usage:
    python -m inference.realtime_pipeline --force "gravity:9.8,wind:2.0" --frames 60
"""

import argparse
import time
from typing import Optional

import torch
import torch.nn as nn

from conditioning.cinematic_controls import (
    CinematicControlEncoder,
    CinematicControls,
    apply_controls,
)
from conditioning.flow_conditioner import FlowConditioner
from physics.neural_continuum_solver import NeuralContinuumSolver


class StubDiffusionModel(nn.Module):
    """
    Placeholder for 4-step distilled diffusion backbone (HunyuanVideo 1.5 / Wan2.2).
    In production, replace with actual diffusion model via hunyuan_adapter or wan2_adapter.
    """

    def __init__(self, conditioning_dim: int = 1280, output_h: int = 480, output_w: int = 832):
        super().__init__()
        self.output_h = output_h
        self.output_w = output_w
        # Minimal decoder simulating 4-step distilled diffusion
        self.decode = nn.Sequential(
            nn.Conv2d(conditioning_dim, 256, 1),
            nn.SiLU(),
            nn.Conv2d(256, 64, 1),
            nn.SiLU(),
            nn.Conv2d(64, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            conditioning: [B, C, h, w] from FlowConditioner
        Returns:
            frame: [B, 3, output_H, output_W]
        """
        x = self.decode(conditioning)
        x = nn.functional.interpolate(x, size=(self.output_h, self.output_w), mode="bilinear", align_corners=False)
        return x


class RealtimePipeline:
    """
    End-to-end real-time generation pipeline.

    Chains: force_vector → NeuralContinuumSolver → FlowConditioner
            → diffusion model → output frame
    """

    TARGET_FPS = 13.2

    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 256,
        output_h: int = 480,
        output_w: int = 832,
        physics_h: int = 60,
        physics_w: int = 104,
        backend: str = "hunyuan",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        conditioning_dim = 1280 if backend == "hunyuan" else 1024

        self.solver = NeuralContinuumSolver(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_h=physics_h,
            output_w=physics_w,
        ).to(self.device)

        self.conditioner = FlowConditioner(
            hidden_dim=hidden_dim,
            conditioning_dim=conditioning_dim,
            backend=backend,
        ).to(self.device)

        self.cinematic_encoder = CinematicControlEncoder(
            conditioning_dim=conditioning_dim,
        ).to(self.device)

        self.diffusion = StubDiffusionModel(
            conditioning_dim=conditioning_dim,
            output_h=output_h,
            output_w=output_w,
        ).to(self.device)

        self.output_h = output_h
        self.output_w = output_w

    def _advance_latent_state(
        self,
        latent_state: torch.Tensor,
        force_vector: torch.Tensor,
        optical_flow: torch.Tensor,
        coarse_rgb: torch.Tensor,
        generated_frame: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the latent state so multi-frame generation is autoregressive.

        The repo does not yet have a trained recurrent world-state model, so we
        evolve the sequence with a deterministic summary of the current physics
        solve and rendered frame. This keeps the prototype honest while avoiding
        the degenerate "same frame repeated N times" behavior.
        """
        latent_dim = latent_state.shape[-1]

        force_summary = self.solver.force_encoder(force_vector)
        flow_summary = optical_flow.mean(dim=(1, 2))
        rgb_summary = coarse_rgb.mean(dim=(1, 2))
        frame_summary = generated_frame.mean(dim=(2, 3))

        summary = torch.cat([flow_summary, rgb_summary, frame_summary], dim=1)
        repeat_factor = (latent_dim + summary.shape[1] - 1) // summary.shape[1]
        summary_latent = summary.repeat(1, repeat_factor)[:, :latent_dim]

        next_token = 0.85 * latent_state[:, -1, :] + 0.15 * (force_summary + 0.05 * summary_latent)
        next_state = torch.roll(latent_state, shifts=-1, dims=1).clone()
        next_state[:, -1, :] = next_token
        return next_state

    @torch.no_grad()
    def run(
        self,
        force_vector: torch.Tensor,
        latent_state: torch.Tensor,
        num_frames: int = 1,
        cinematic: Optional[CinematicControls] = None,
    ) -> list[torch.Tensor]:
        """
        Generate frames at target 13.2 FPS.

        Args:
            force_vector: [B, 6] force + torque
            latent_state: [B, seq_len, latent_dim] initial scene state
            num_frames: Number of frames to generate
            cinematic: Optional cinematic control parameters
        Returns:
            List of frame tensors [B, 3, H, W]
        """
        force_vector = force_vector.to(self.device)
        latent_state = latent_state.to(self.device)

        # Pre-compute cinematic conditioning if provided
        cine_cond = None
        if cinematic is not None:
            cine_cond = apply_controls(cinematic, self.cinematic_encoder, self.device)

        frames = []
        for _ in range(num_frames):
            # Physics solver → optical flow + coarse RGB
            optical_flow, coarse_rgb = self.solver(latent_state, force_vector)

            # Flow conditioning for diffusion
            conditioning = self.conditioner(optical_flow, coarse_rgb)

            # Add cinematic controls if available
            if cine_cond is not None:
                # Broadcast cinematic vector across spatial dims
                cine_spatial = cine_cond.unsqueeze(-1).unsqueeze(-1)
                cine_spatial = cine_spatial.expand_as(conditioning)
                conditioning = conditioning + cine_spatial

            # Diffusion model generates final frame
            frame = self.diffusion(conditioning)
            frames.append(frame)

            # Advance the latent sequence so subsequent frames respond to the
            # previous generation instead of replaying the same state forever.
            latent_state = self._advance_latent_state(
                latent_state=latent_state,
                force_vector=force_vector,
                optical_flow=optical_flow,
                coarse_rgb=coarse_rgb,
                generated_frame=frame,
            )

        return frames

    @torch.no_grad()
    def benchmark(
        self,
        num_frames: int = 50,
        batch_size: int = 1,
        seq_len: int = 64,
        warmup: int = 10,
    ) -> dict:
        """
        Measure actual FPS and report vs 13.2 FPS target.

        Returns:
            Dict with fps, avg_frame_ms, target_fps, meets_target
        """
        latent_dim = self.solver.latent_dim
        force = torch.randn(batch_size, 6, device=self.device)
        state = torch.randn(batch_size, seq_len, latent_dim, device=self.device)

        # Warmup
        for _ in range(warmup):
            self.run(force, state, num_frames=1)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        self.run(force, state, num_frames=num_frames)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / num_frames) * 1000
        fps = num_frames / elapsed

        return {
            "fps": fps,
            "avg_frame_ms": avg_ms,
            "target_fps": self.TARGET_FPS,
            "meets_target": fps >= self.TARGET_FPS,
            "num_frames": num_frames,
        }


def parse_force_string(force_str: str) -> torch.Tensor:
    """Parse 'gravity:9.8,wind:2.0' into a [1, 6] tensor."""
    force = torch.zeros(1, 6)
    for item in force_str.split(","):
        key, value_str = item.split(":")
        value = float(value_str)
        key = key.strip().lower()
        if key == "gravity":
            force[0, 1] = -value  # negative y
        elif key == "wind":
            force[0, 0] = value
        elif key == "torque":
            force[0, 3] = value
    return force


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time cinematic generation pipeline")
    parser.add_argument("--force", type=str, default="gravity:9.8", help="Force spec, e.g. gravity:9.8,wind:2.0")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to generate")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark instead of generation")
    parser.add_argument("--backend", type=str, default="hunyuan", choices=["hunyuan", "wan2"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    pipeline = RealtimePipeline(backend=args.backend, device=args.device)

    if args.benchmark:
        results = pipeline.benchmark(num_frames=args.frames)
        print(f"FPS: {results['fps']:.1f} (target: {results['target_fps']})")
        print(f"Avg frame time: {results['avg_frame_ms']:.1f}ms")
        print(f"Meets target: {'YES' if results['meets_target'] else 'NO'}")
    else:
        force = parse_force_string(args.force)
        state = torch.randn(1, 64, 512)
        frames = pipeline.run(force, state, num_frames=args.frames)
        print(f"Generated {len(frames)} frames, shape: {frames[0].shape}")


if __name__ == "__main__":
    main()
