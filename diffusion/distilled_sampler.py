"""
Distilled Sampler

Few-step sampler supporting 1, 2, or 4-step generation via
consistency distillation. Enables real-time inference by reducing
the typical 20-50 diffusion steps to 1-4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistilledSampler(nn.Module):
    """
    Consistency-distilled sampler for few-step diffusion generation.

    Supports 1, 2, and 4-step sampling schedules. Uses a learned
    denoising network that predicts clean samples directly.
    """

    def __init__(
        self,
        channels: int = 1280,
        num_refinement_blocks: int = 3,
    ):
        super().__init__()
        self.channels = channels

        # Noise schedule endpoints for 1/2/4 steps
        self.register_buffer(
            "sigma_schedule_4",
            torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]),
        )
        self.register_buffer(
            "sigma_schedule_2",
            torch.tensor([1.0, 0.5, 0.0]),
        )
        self.register_buffer(
            "sigma_schedule_1",
            torch.tensor([1.0, 0.0]),
        )

        # Refinement network for direct prediction
        layers = []
        for _ in range(num_refinement_blocks):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.GroupNorm(32, channels),
                    nn.SiLU(),
                ]
            )
        layers.append(nn.Conv2d(channels, channels, 1))
        self.refine_net = nn.Sequential(*layers)

        # Sigma embedding
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    @staticmethod
    def _infer_timestep_dim(model: nn.Module, default: int = 256) -> int:
        """Infer the timestep embedding width expected by the diffusion model."""
        time_embed = getattr(model, "time_embed", None)
        if time_embed is None:
            return default
        for module in time_embed.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        return default

    def _get_schedule(self, num_steps: int) -> torch.Tensor:
        if num_steps == 1:
            return self.sigma_schedule_1
        elif num_steps == 2:
            return self.sigma_schedule_2
        elif num_steps == 4:
            return self.sigma_schedule_4
        else:
            # Linear schedule for arbitrary step counts
            return torch.linspace(1.0, 0.0, num_steps + 1, device=self.sigma_schedule_1.device)

    def _denoise_step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        sigma: float,
        sigma_next: float,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step."""
        B = x.shape[0]
        sigma_t = torch.full((B, 1), sigma, device=x.device)
        sigma_emb = self.sigma_embed(sigma_t)[:, :, None, None]

        # Model prediction with sigma conditioning
        x_cond = x + sigma_emb
        timestep_dim = self._infer_timestep_dim(model)
        pred = model(x_cond, cond, sigma_t.expand(B, timestep_dim))

        # Linear interpolation toward clean sample
        alpha = 1.0 - sigma_next / max(sigma, 1e-8)
        x_next = x + alpha * (pred - x)

        return x_next

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        x_T: torch.Tensor,
        cond: torch.Tensor,
        num_steps: int = 4,
    ) -> torch.Tensor:
        """
        Few-step sampling.

        Args:
            model: Diffusion model (e.g., HunyuanVideoAdapter or Wan2Adapter)
            x_T: [B, C, H, W] initial noise
            cond: [B, C_cond, h, w] conditioning tensor
            num_steps: Number of denoising steps (1, 2, or 4)
        Returns:
            x_0: [B, C, H, W] denoised output
        """
        schedule = self._get_schedule(num_steps)
        x = x_T

        for i in range(num_steps):
            sigma = schedule[i].item()
            sigma_next = schedule[i + 1].item()
            x = self._denoise_step(model, x, sigma, sigma_next, cond)

        # Final refinement pass
        x = x + self.refine_net(x)

        return x

    def consistency_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Consistency distillation loss.

        Ensures that sampling with fewer steps produces results
        consistent with more steps.

        Args:
            model: Diffusion model
            x_0: [B, C, H, W] clean target
            cond: [B, C_cond, h, w] conditioning
        Returns:
            loss: Scalar consistency loss
        """
        B = x_0.shape[0]
        noise = torch.randn_like(x_0)

        # Noisy sample at random sigma
        sigma = torch.rand(B, 1, 1, 1, device=x_0.device)
        x_noisy = x_0 + sigma * noise

        # 4-step prediction (teacher)
        with torch.no_grad():
            pred_4 = self.sample(model, x_noisy, cond, num_steps=4)

        # 1-step prediction (student) — must enable gradients since self.sample() uses @torch.no_grad()
        with torch.enable_grad():
            pred_1 = self.sample(model, x_noisy, cond, num_steps=1)

        # Consistency loss: 1-step should match 4-step
        loss = F.mse_loss(pred_1, pred_4.detach())

        return loss
