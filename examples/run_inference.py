from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from inference.realtime_pipeline import RealtimePipeline


def main() -> None:
    pipeline = RealtimePipeline(
        latent_dim=128,
        hidden_dim=64,
        output_h=128,
        output_w=128,
        physics_h=16,
        physics_w=16,
        device="cpu",
    )
    force = torch.tensor([[1.5, -9.8, 0.0, 0.1, 0.0, 0.0]], dtype=torch.float32)
    state = torch.randn(1, 8, 128)

    frames = pipeline.run(force, state, num_frames=1)
    frame = frames[0][0].permute(1, 2, 0).cpu().numpy()
    frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

    output_dir = Path("examples") / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "random_inference.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))

    print("Saved frame to", output_path)
    print("Frame shape:", tuple(frame_u8.shape))


if __name__ == "__main__":
    main()
