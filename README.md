# cinematic-controlnet

**Action-Conditioned Cinematic Bridge via Neural Continuum Mechanics**

A real-time action-conditioned video generation framework implementing RealWonder (Liu et al., Stanford/USC, March 2026) with a critical architectural upgrade: the offline Blender physics engine is replaced with a real-time learned neural continuum mechanics solver operating in the latent space.

The thesis: 2D conditioning alone is not enough for director-facing video tools. Creative teams need controls that connect camera language, motion, force, scene structure, and multi-shot continuity.

---

## Portfolio Context

This repo is part of [Creative AI Workflows](https://chrismado.github.io/creative-ai-workflows/) ([source](https://github.com/chrismado/creative-ai-workflows)), a portfolio showcase connecting generative video, 3D scene review, creative QA, and enterprise deployment.

In that system, `cinematic-controlnet` is the **cinematic control layer**. It explores how a director's intent can become model controls: camera path, depth, motion, physical force, style cues, and continuity constraints.

### Customer-Facing Use Case

A creative team wants to move beyond one-off prompt results and produce repeatable shot variants from the same brief. This repo is positioned as a workflow for turning direction into controllable generation, then reviewing variants in a way that supports real creative feedback.

### Demo Narrative

- Start with a director-style brief: subject, camera, motion, tone, and continuity target.
- Show how that brief becomes control inputs rather than only a text prompt.
- Generate variants, compare them, and explain which version best preserves creative intent.

---

## Why 2D Control Is Not Enough

Standard ControlNet (lllyasviel/controlnet) and its video adaptations (control-a-video, ControlVideo) operate entirely in 2D screen space. They extract condition maps from source videos and guide generation — but they have no structural understanding of 3D scenes and cannot interpret how continuous 3D forces propagate through an environment.

The result: physically plausible-looking outputs that violate real physics. Objects drift, forces don't propagate, fluid dynamics are wrong.

**RealWonder (March 2026) solved this** by using a physics simulation engine as an intermediate bridge — translating physical actions through a simulator into dense optical flow matrices, which then condition a 4-step distilled diffusion model. The result: 13.2 FPS real-time generation with correct Newtonian mechanics.

**The problem with Blender:** Using Blender as the physics engine introduces 200-400ms rendering latency per frame, breaking the real-time claim. For interactive directors tools and live video modeling, this is a hard architectural failure.

**This repo's solution:** Replace Blender with a lightweight learned neural dynamics model that approximates physical forces directly within the latent space — enabling the live real-time feedback loops required for interactive cinematic production.

---

## Architecture

```
Director Input
(3D force vector / camera pose / object injection)
        │
        ▼
┌────────────────────────────────┐
│   Neural Continuum Mechanics   │  ← KEY UPGRADE
│   Solver (latent space)        │    Replaces offline Blender
│   Approximates: rigid body,    │    ~8ms inference vs ~300ms
│   fluid dynamics, cloth,       │
│   granular materials           │
└─────────────┬──────────────────┘
              │
     Dense Optical Flow Matrices
     (3D velocity field → 2D image plane)
              │
     Coarse RGB Representations
     (structural cues, occlusion changes)
              │
              ▼
┌────────────────────────────────┐
│   4-Step Distilled Diffusion   │
│   HunyuanVideo 1.5 / Wan2.2   │
│   Conditioned on flow + RGB    │
└─────────────┬──────────────────┘
              │
              ▼
     Cinematographic Control Layer
     (lens specs · depth maps · LUTs)
              │
              ▼
     Final Video Output
     13.2 FPS · 480x832
```

### Cinematographic Control Signals
Unlike standard ControlNet which only accepts depth/canny maps, this pipeline accepts filmmaker-native control signals:
- Camera lens specifications (focal length, aperture)
- Depth maps with artistic intent
- Color grading LUTs as action inputs
- Multi-shot narrative continuity constraints

---

## Stack

- **Neural physics:** Custom lightweight continuum mechanics solver (latent space)
- **Optical flow:** Dense flow matrices via PWC-Net adapted for latent projection
- **Diffusion backbone:** HunyuanVideo 1.5 or Wan2.2 (4-step distilled)
- **Training hardware:** RTX 4090 (diffusion backbone) + RTX 3090 (physics solver)
- **Framework:** PyTorch, CUDA custom kernels

---

## Benchmarks

Median results from three fresh runs of `python -m inference.benchmark --compare-blender --frames 20`:

| Metric | Measured Value |
|--------|----------------|
| Neural physics solver latency | 1.81 ms |
| Flow conditioner latency | 0.51 ms |
| Diffusion model latency | 0.23 ms |
| Total staged latency | 2.53 ms |
| Effective staged FPS | 394.6 |
| End-to-end frame time | 2.96 ms |
| End-to-end FPS | 337.9 |
| Simulated Blender stage latency | 18.86 ms |
| Solver speedup vs simulated Blender | 10.3x |
| Estimated Blender pipeline FPS | 51.1 |

Benchmarks measured on an NVIDIA GeForce RTX 3090 with PyTorch 2.11.0+cu128 and
Python 3.12.2 on April 9, 2026. The Blender comparison is a CPU-side synthetic
baseline from the benchmark script, not a live Blender render.

See `PERFORMANCE.md` for the profiler breakdown and `torch.compile()` results.

---

## Directory Structure

```
cinematic-controlnet/
├── conditioning/
│   ├── __init__.py
│   ├── cinematic_controls.py
│   ├── flow_conditioner.py
│   └── multi_shot_consistency.py
├── diffusion/
│   ├── __init__.py
│   ├── distilled_sampler.py
│   ├── hunyuan_adapter.py
│   └── wan2_adapter.py
├── inference/
│   ├── __init__.py
│   ├── benchmark.py
│   └── realtime_pipeline.py
├── physics/
│   ├── __init__.py
│   ├── flow_generator.py
│   ├── force_tokenizer.py
│   ├── neural_continuum_solver.py
│   └── rgb_renderer.py
├── tests/
│   ├── __init__.py
│   └── test_physics_accuracy.py
├── training/
│   ├── __init__.py
│   ├── train_conditioner.py
│   └── train_physics_solver.py
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/chrismado/cinematic-controlnet
cd cinematic-controlnet
pip install -r requirements.txt

# Run interactive demo
python -m inference.realtime_pipeline --force "gravity:9.8,wind:2.0" --frames 60

# Benchmark against RealWonder baseline
python -m inference.benchmark --compare-blender

# Train neural physics solver
python -m training.train_physics_solver --dataset physion --epochs 100
```

---

## References

1. **RealWonder: Real-Time Physical Action-Conditioned Video Generation** — Liu, Chen, Li, Wang, Yu, Wu (Stanford University / USC), March 2026. arxiv 2603.05449. Core architecture this repo implements.
2. **PerpetualWonder** — Long-horizon context collapse solution for continuous generation.
3. **ActionParty** — Multi-subject action binding in generative video.
4. **GenReward** — Video diffusion as RL reward model.
5. **Weifeng-Chen/control-a-video** — 2D screen-space ControlNet for video (legacy approach this supersedes).
6. **YBYBZhang/ControlVideo** — ICLR 2024 training-free ControlNet (legacy approach this supersedes).
7. **lllyasviel/controlnet** — Original ControlNet (legacy image-space architecture).
8. **Wan2.2** — Open-source diffusion backbone used as generation model.
9. **HunyuanVideo 1.5** — Alternative diffusion backbone.

---

*Targeting Higgsfield AI (#1 signal), Runway ML GWM-1 team, Decart live video modeling roles.*
