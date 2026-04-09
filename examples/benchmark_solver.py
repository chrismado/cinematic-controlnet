from __future__ import annotations

import time

import torch

from physics.neural_continuum_solver import NeuralContinuumSolver


def benchmark_resolution(output_hw: tuple[int, int], iterations: int = 10) -> float:
    height, width = output_hw
    model = NeuralContinuumSolver(latent_dim=128, hidden_dim=64, output_h=height, output_w=width).cpu().eval()
    latent = torch.randn(1, 8, 128)
    force = torch.randn(1, 6)

    with torch.no_grad():
        for _ in range(2):
            model(latent, force)

        start = time.perf_counter()
        for _ in range(iterations):
            model(latent, force)
        elapsed = time.perf_counter() - start

    return elapsed * 1000 / iterations


def main() -> None:
    print("| Resolution | Avg latency (ms) |")
    print("|------------|------------------|")
    for resolution in [(16, 16), (32, 32), (48, 64)]:
        avg_ms = benchmark_resolution(resolution)
        print(f"| {resolution[0]}x{resolution[1]} | {avg_ms:>16.2f} |")


if __name__ == "__main__":
    main()
