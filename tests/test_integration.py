from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from conditioning.flow_conditioner import FlowConditioner
from physics.flow_generator import OpticalFlowGenerator
from physics.neural_continuum_solver import NeuralContinuumSolver
from training.train_physics_solver import create_synthetic_dataset

pytestmark = pytest.mark.integration


def test_physics_to_conditioning_pipeline_runs_on_cpu() -> None:
    solver = NeuralContinuumSolver(latent_dim=64, hidden_dim=32, output_h=16, output_w=16).cpu().eval()
    generator = (
        OpticalFlowGenerator(
            state_dim=64,
            base_channels=16,
            output_h=16,
            output_w=16,
        )
        .cpu()
        .eval()
    )
    conditioner = (
        FlowConditioner(
            hidden_dim=32,
            conditioning_dim=128,
            backend="hunyuan",
        )
        .cpu()
        .eval()
    )

    latent = torch.randn(2, 8, 64)
    force = torch.randn(2, 6)

    with torch.no_grad():
        solver_flow, coarse_rgb = solver(latent, force)
        physics_state = solver.force_encoder(force)
        generated_flow = generator(physics_state, coarse_rgb.permute(0, 3, 1, 2))
        conditioning = conditioner(generated_flow.permute(0, 2, 3, 1), coarse_rgb)

    assert solver_flow.shape == (2, 16, 16, 2)
    assert coarse_rgb.shape == (2, 16, 16, 3)
    assert generated_flow.shape == (2, 2, 16, 16)
    assert conditioning.shape[0] == 2
    assert conditioning.shape[1] == 128


def test_training_loop_runs_two_steps_on_synthetic_data(tmp_path: Path) -> None:
    dataset = create_synthetic_dataset(
        num_samples=4,
        seq_len=8,
        latent_dim=64,
        output_h=8,
        output_w=8,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    model = NeuralContinuumSolver(latent_dim=64, hidden_dim=32, output_h=8, output_w=8).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses: list[float] = []
    for step, (latent, force, target_flow, target_rgb) in enumerate(loader):
        pred_flow, pred_rgb = model(latent, force)
        loss = torch.nn.functional.mse_loss(pred_flow, target_flow) + torch.nn.functional.mse_loss(pred_rgb, target_rgb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step == 1:
            break

    assert len(losses) == 2
    assert all(loss >= 0 for loss in losses)
