"""Profile the neural continuum solver and compare eager vs compiled execution."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import sysconfig
import time
from pathlib import Path
from typing import Iterable

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.neural_continuum_solver import NeuralContinuumSolver  # noqa: E402


def _ensure_stdlib_profile_module() -> None:
    if "profile" in sys.modules:
        return
    profile_path = Path(sysconfig.get_path("stdlib")) / "profile.py"
    spec = importlib.util.spec_from_file_location("profile", profile_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["profile"] = module
    spec.loader.exec_module(module)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_latency(
    model: NeuralContinuumSolver,
    latent: torch.Tensor,
    force: torch.Tensor,
    warmup: int,
    iters: int,
) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            model(latent, force)
        _sync(latent.device)
        start = time.perf_counter()
        for _ in range(iters):
            model(latent, force)
        _sync(latent.device)
    return (time.perf_counter() - start) * 1000.0 / iters


def _activities(device: torch.device) -> list[ProfilerActivity]:
    acts = [ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    return acts


def _sort_key(device: torch.device) -> str:
    return "cuda_time_total" if device.type == "cuda" else "cpu_time_total"


def _compile_model(model: NeuralContinuumSolver, enabled: bool) -> NeuralContinuumSolver:
    if not enabled or not hasattr(torch, "compile"):
        return model
    _ensure_stdlib_profile_module()
    return torch.compile(model, mode="reduce-overhead")  # type: ignore[return-value]


def _max_abs_diff(lhs: Iterable[torch.Tensor], rhs: Iterable[torch.Tensor]) -> float:
    return max((a - b).abs().max().item() for a, b in zip(lhs, rhs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile NeuralContinuumSolver")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--profile-iters", type=int, default=10)
    parser.add_argument("--disable-compile", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available() and args.device == "cuda":
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")

    torch.manual_seed(0)

    model = NeuralContinuumSolver().to(device).eval()
    latent = torch.randn(1, args.seq_len, 512, device=device)
    force = torch.randn(1, 6, device=device)

    eager_ms = _measure_latency(model, latent, force, args.warmup, args.iters)
    with torch.no_grad():
        ref = model(latent, force)

    compiled_ms = None
    max_diff = None
    if not args.disable_compile:
        compiled = _compile_model(model, enabled=True)
        with torch.no_grad():
            out = compiled(latent, force)
        max_diff = _max_abs_diff(ref, out)
        compiled_ms = _measure_latency(compiled, latent, force, args.warmup, args.iters)

    with profile(activities=_activities(device), record_shapes=True) as prof:
        with torch.no_grad():
            for _ in range(args.profile_iters):
                model(latent, force)

    print(f"Device: {device}")
    print(f"Eager latency: {eager_ms:.2f} ms")
    if compiled_ms is not None:
        speedup = eager_ms / compiled_ms if compiled_ms > 0 else float("inf")
        print(f"Compiled latency: {compiled_ms:.2f} ms")
        print(f"Compile speedup: {speedup:.2f}x")
        print(f"Max abs diff: {max_diff:.3e}")
    else:
        print("Compiled latency: skipped")
    print()
    print(
        prof.key_averages().table(
            sort_by=_sort_key(device),
            row_limit=10,
        )
    )


if __name__ == "__main__":
    main()
