# cinematic-controlnet

**Action-Conditioned Cinematic Control Architecture Prototype**

An architecture prototype inspired by action-conditioned video generation systems such as RealWonder. The repo explores how an offline simulator bridge could be replaced by a lightweight learned dynamics module that exposes director-friendly controls in the latent space.

The thesis: 2D conditioning alone is not enough for director-facing video tools. Creative teams need controls that connect camera language, motion, force, scene structure, and multi-shot continuity.

---

## Current Status

This is an architecture prototype, not a trained production video model.

- The default neural modules are untrained and randomly initialized unless you wire in real checkpoints.
- The repo demonstrates the control interface, conditioning data flow, benchmark harness, and integration points.
- It does not prove trained physics simulation quality, production video generation quality, or real Blender replacement quality.
- The benchmark numbers measure staged prototype module latency and a synthetic Blender proxy, not end-to-end quality on a real simulator or diffusion backbone.

The honest value of this repo is the workflow design: how director intent could move from force, camera, lens, depth, style, and continuity inputs into a controllable generation pipeline.

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

RealWonder-style systems address this by using a physics simulation engine as an intermediate bridge, translating physical actions through a simulator into dense optical flow matrices that condition a distilled diffusion model.

**The problem with Blender:** Using Blender as the physics engine introduces 200-400ms rendering latency per frame, breaking the real-time claim. For interactive directors tools and live video modeling, this is a hard architectural failure.

**This repo's design target:** Explore whether a lightweight learned dynamics model could approximate physical force responses directly within the latent space, enabling the live feedback loops required for interactive cinematic production.

---

## Architecture

```
Director Input
(3D force vector / camera pose / object injection)
        │
        ▼
┌────────────────────────────────┐
│   Neural Continuum Mechanics   │  ← DESIGN TARGET
│   Solver (latent space)        │    Candidate Blender substitute
│   Approximates: rigid body,    │    Target: low-latency inference
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

- **Neural physics:** Prototype continuum mechanics module in latent space
- **Optical flow:** Dense flow matrices via PWC-Net adapted for latent projection
- **Diffusion backbone:** Adapter interface for HunyuanVideo / Wan-style backbones
- **Training status:** Untrained by default; real checkpoints must be supplied separately
- **Framework:** PyTorch with CUDA-capable execution when available

---

## Benchmarks

Latency smoke-test results from three runs of `python -m inference.benchmark --compare-blender --frames 20`:

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
baseline from the benchmark script, not a live Blender render. These numbers are
prototype latency measurements, not trained-model quality measurements.

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

Current test files: `tests/test_integration.py` and `tests/test_module_shapes.py`.

---

## Quick Start

```bash
git clone https://github.com/chrismado/cinematic-controlnet
cd cinematic-controlnet
pip install -r requirements.txt

# Run interactive demo
python -m inference.realtime_pipeline --force "gravity:9.8,wind:2.0" --frames 60

# Benchmark staged prototype latency
python -m inference.benchmark --compare-blender

# Run prototype training script
python -m training.train_physics_solver --dataset physion --epochs 100
```

The training scripts and smoke tests exercise the prototype modules, but the
default path still assumes synthetic or placeholder data unless you wire in
real datasets and checkpoints.

---

## References

1. **RealWonder: Real-Time Physical Action-Conditioned Video Generation** — Liu, Chen, Li, Wang, Yu, Wu (Stanford University / USC), March 2026. arxiv 2603.05449. Architecture reference for this prototype.
2. **PerpetualWonder** — Long-horizon context collapse solution for continuous generation.
3. **ActionParty** — Multi-subject action binding in generative video.
4. **GenReward** — Video diffusion as RL reward model.
5. **Weifeng-Chen/control-a-video** — 2D screen-space ControlNet for video (legacy approach this supersedes).
6. **YBYBZhang/ControlVideo** — ICLR 2024 training-free ControlNet (legacy approach this supersedes).
7. **lllyasviel/controlnet** — Original ControlNet (legacy image-space architecture).
8. **Wan2.2** — Open-source diffusion backbone family for adapter exploration.
9. **HunyuanVideo 1.5** — Alternative diffusion backbone family for adapter exploration.

---

*Targeting Higgsfield AI (#1 signal), Runway ML GWM-1 team, Decart live video modeling roles.*
