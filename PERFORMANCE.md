# Performance Profiling

This repo includes a reproducible profiler entry point in `scripts/profile.py` for the
`NeuralContinuumSolver`.

## Measurement Setup

- Date: April 9, 2026
- Hardware: NVIDIA GeForce RTX 3090
- Python: 3.12.2
- PyTorch: 2.11.0+cu128
- Command: `python scripts/profile.py --device cuda --seq-len 64 --warmup 5 --iters 20 --profile-iters 10`

## Bottlenecks

The top CUDA-heavy operator groups in the solver profile were:

1. `aten::addmm` / GEMM-backed linear projections: 3.87 ms CUDA total across 10 profiled forwards
2. `aten::_transformer_encoder_layer_fwd`: 3.15 ms CUDA total
3. `aten::_native_multi_head_attention` and `aten::scaled_dot_product_attention`: 1.46 ms and 0.64 ms CUDA total

These numbers point to the expected hotspot pattern: most of the time is spent in
linear projections around the transformer block, then in the encoder block itself,
with attention kernels contributing a smaller but still visible share.

## Optimizations Applied

- Added `scripts/profile.py` so the eager and compiled paths can be profiled with the same weights and inputs.
- The profiling path enables `torch.set_float32_matmul_precision("high")` on CUDA to let Tensor Core-backed matmuls use the faster high-precision mode.
- `torch.compile(mode="reduce-overhead")` was validated on the same model instance and input batch.

## Before / After

| Mode | Avg Latency (ms) | Speedup | Max Abs Diff |
|------|------------------|---------|--------------|
| Eager | 1.22 | 1.00x | - |
| `torch.compile()` | 0.82 | 1.49x | `9.54e-05` |

The compiled path preserved outputs within floating-point noise and materially improved
steady-state latency. I did not wire compiled mode into the default CLI path because the
first-run graph capture cost is better amortized in long-lived services than in short
benchmark or demo runs.

## Recommendations

- Use `torch.compile()` in persistent GPU inference services where the solver is reused for many requests.
- Keep solver batch sizes stable when possible so the compiled graph stays hot.
- If end-to-end latency matters more than throughput, focus optimization work on the linear projection stack first, then on transformer block width and sequence length.
