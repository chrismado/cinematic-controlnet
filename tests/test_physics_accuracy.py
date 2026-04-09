"""
Unit Tests for All Modules

Tests all modules on CPU with random tensors to verify:
  - Forward pass shapes are correct
  - No runtime errors with expected inputs
  - Gradient flow works for trainable modules
"""

import unittest

import torch

from conditioning.cinematic_controls import (
    CinematicControlEncoder,
    CinematicControls,
    apply_controls,
)
from conditioning.flow_conditioner import FlowConditioner
from conditioning.multi_shot_consistency import MultiShotConsistency
from diffusion.distilled_sampler import DistilledSampler
from diffusion.hunyuan_adapter import HunyuanVideoAdapter
from diffusion.wan2_adapter import Wan2Adapter
from physics.flow_generator import OpticalFlowGenerator, flow_to_rgb
from physics.force_tokenizer import ForceTokenizer
from physics.neural_continuum_solver import NeuralContinuumSolver
from physics.rgb_renderer import LatentRGBRenderer


class TestNeuralContinuumSolver(unittest.TestCase):
    def setUp(self):
        self.model = NeuralContinuumSolver(latent_dim=64, hidden_dim=32, output_h=8, output_w=8)

    def test_forward_shape(self):
        state = torch.randn(2, 4, 64)
        force = torch.randn(2, 6)
        flow, rgb = self.model(state, force)
        self.assertEqual(flow.shape, (2, 8, 8, 2))
        self.assertEqual(rgb.shape, (2, 8, 8, 3))

    def test_gradient_flow(self):
        state = torch.randn(1, 4, 64, requires_grad=True)
        force = torch.randn(1, 6)
        flow, rgb = self.model(state, force)
        loss = flow.sum() + rgb.sum()
        loss.backward()
        self.assertIsNotNone(state.grad)


class TestOpticalFlowGenerator(unittest.TestCase):
    def setUp(self):
        self.model = OpticalFlowGenerator(
            state_dim=64,
            prev_frame_channels=3,
            base_channels=16,
            output_h=16,
            output_w=16,
        )

    def test_forward_shape(self):
        state = torch.randn(2, 64)
        prev = torch.randn(2, 3, 16, 16)
        flow = self.model(state, prev)
        self.assertEqual(flow.shape, (2, 2, 16, 16))

    def test_flow_to_rgb(self):
        flow = torch.randn(2, 2, 8, 8)
        rgb = flow_to_rgb(flow)
        self.assertEqual(rgb.shape, (2, 3, 8, 8))
        self.assertTrue(rgb.min() >= 0.0)
        self.assertTrue(rgb.max() <= 1.0)


class TestLatentRGBRenderer(unittest.TestCase):
    def setUp(self):
        self.model = LatentRGBRenderer(
            latent_dim=64,
            hidden_dim=32,
            output_h=32,
            output_w=32,
        )

    def test_forward_shape(self):
        latent = torch.randn(2, 64)
        flow = torch.randn(2, 2, 32, 32)
        rgb = self.model(latent, flow)
        self.assertEqual(rgb.shape, (2, 3, 32, 32))
        # Output should be in [0, 1] due to Sigmoid
        self.assertTrue(rgb.min() >= 0.0)
        self.assertTrue(rgb.max() <= 1.0)


class TestForceTokenizer(unittest.TestCase):
    def setUp(self):
        self.model = ForceTokenizer(
            force_channels=6,
            hidden_dim=32,
            codebook_size=64,
            codebook_dim=16,
        )

    def test_forward_shape(self):
        field = torch.randn(2, 6, 16, 16)
        tokens, loss = self.model(field)
        self.assertEqual(tokens.dim(), 3)  # [B, h, w]
        self.assertEqual(tokens.shape[0], 2)
        self.assertTrue(loss.dim() == 0)  # scalar

    def test_decode_shape(self):
        field = torch.randn(2, 6, 16, 16)
        tokens, _ = self.model(field)
        reconstructed = self.model.decode(tokens)
        self.assertEqual(reconstructed.shape[0], 2)
        self.assertEqual(reconstructed.shape[1], 6)

    def test_encode_and_decode(self):
        field = torch.randn(2, 6, 16, 16)
        recon, loss = self.model.encode_and_decode(field)
        self.assertEqual(recon.shape[0], 2)
        self.assertEqual(recon.shape[1], 6)


class TestFlowConditioner(unittest.TestCase):
    def test_hunyuan_backend(self):
        model = FlowConditioner(hidden_dim=32, conditioning_dim=64, backend="hunyuan")
        flow = torch.randn(2, 16, 16, 2)
        rgb = torch.randn(2, 16, 16, 3)
        out = model(flow, rgb)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 64)

    def test_wan2_backend(self):
        model = FlowConditioner(hidden_dim=32, conditioning_dim=48, backend="wan2")
        flow = torch.randn(2, 16, 16, 2)
        rgb = torch.randn(2, 16, 16, 3)
        out = model(flow, rgb)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 48)


class TestCinematicControls(unittest.TestCase):
    def test_camera_encoding(self):
        encoder = CinematicControlEncoder(conditioning_dim=64)
        controls = CinematicControls(focal_length=85.0, aperture=1.4)
        device = torch.device("cpu")
        result = encoder(controls, device)
        self.assertIn("camera", result)
        self.assertEqual(result["camera"].shape, (1, 64))

    def test_with_depth_map(self):
        encoder = CinematicControlEncoder(conditioning_dim=64)
        controls = CinematicControls(depth_map=torch.randn(16, 16))
        device = torch.device("cpu")
        result = encoder(controls, device)
        self.assertIn("depth", result)

    def test_apply_controls(self):
        encoder = CinematicControlEncoder(conditioning_dim=64)
        controls = CinematicControls()
        device = torch.device("cpu")
        combined = apply_controls(controls, encoder, device)
        self.assertEqual(combined.shape, (1, 64))


class TestMultiShotConsistency(unittest.TestCase):
    def setUp(self):
        self.model = MultiShotConsistency(
            feature_dim=64,
            num_heads=4,
            num_layers=2,
            metadata_dim=8,
        )

    def test_forward_shape(self):
        shots = [torch.randn(2, 4, 64) for _ in range(3)]
        meta = [torch.randn(2, 8) for _ in range(3)]
        outputs = self.model(shots, meta)
        self.assertEqual(len(outputs), 3)
        for out in outputs:
            self.assertEqual(out.shape, (2, 4, 64))

    def test_consistency_loss(self):
        shots = [torch.randn(2, 4, 64) for _ in range(3)]
        meta = [torch.randn(2, 8) for _ in range(3)]
        loss = self.model.consistency_loss(shots, meta)
        self.assertTrue(loss.dim() == 0)
        self.assertTrue(loss.item() >= 0.0)


class TestHunyuanVideoAdapter(unittest.TestCase):
    def setUp(self):
        self.model = HunyuanVideoAdapter(
            model_channels=64,
            cond_channels=64,
            num_blocks=2,
            timestep_dim=32,
        )

    def test_forward_shape(self):
        x = torch.randn(2, 64, 8, 8)
        cond = torch.randn(2, 64, 8, 8)
        t = torch.randn(2, 32)
        out = self.model(x, cond, t)
        self.assertEqual(out.shape, (2, 64, 8, 8))

    def test_zero_init(self):
        """Verify zero-conv output is zero at initialization."""
        x = torch.randn(1, 64, 4, 4)
        cond = torch.randn(1, 64, 4, 4)
        t = torch.randn(1, 32)
        out = self.model(x, cond, t)
        # Output should be close to zero due to zero-conv init
        self.assertTrue(out.abs().max().item() < 1e-5)


class TestWan2Adapter(unittest.TestCase):
    def setUp(self):
        self.model = Wan2Adapter(
            model_channels=64,
            cond_channels=64,
            num_blocks=2,
            timestep_dim=32,
        )

    def test_forward_shape(self):
        x = torch.randn(2, 64, 8, 8)
        cond = torch.randn(2, 64, 8, 8)
        t = torch.randn(2, 32)
        out = self.model(x, cond, t)
        self.assertEqual(out.shape, (2, 64, 8, 8))

    def test_zero_init(self):
        x = torch.randn(1, 64, 4, 4)
        cond = torch.randn(1, 64, 4, 4)
        t = torch.randn(1, 32)
        out = self.model(x, cond, t)
        self.assertTrue(out.abs().max().item() < 1e-5)


class TestDistilledSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = DistilledSampler(channels=64, num_refinement_blocks=2)
        self.model = HunyuanVideoAdapter(
            model_channels=64,
            cond_channels=64,
            num_blocks=2,
            timestep_dim=32,
        )

    def test_sample_4_step(self):
        x_T = torch.randn(1, 64, 4, 4)
        cond = torch.randn(1, 64, 4, 4)
        out = self.sampler.sample(self.model, x_T, cond, num_steps=4)
        self.assertEqual(out.shape, (1, 64, 4, 4))

    def test_sample_2_step(self):
        x_T = torch.randn(1, 64, 4, 4)
        cond = torch.randn(1, 64, 4, 4)
        out = self.sampler.sample(self.model, x_T, cond, num_steps=2)
        self.assertEqual(out.shape, (1, 64, 4, 4))

    def test_sample_1_step(self):
        x_T = torch.randn(1, 64, 4, 4)
        cond = torch.randn(1, 64, 4, 4)
        out = self.sampler.sample(self.model, x_T, cond, num_steps=1)
        self.assertEqual(out.shape, (1, 64, 4, 4))


if __name__ == "__main__":
    unittest.main()
