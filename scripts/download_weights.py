from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_PRESETS: dict[str, list[dict[str, str]]] = {
    "small": [
        {
            "label": "Stable Video Diffusion config",
            "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt",
            "filename": "model_index.json",
            "revision": "9e43909513c6714f1bc78bcb44d96e733cd242aa",
            "subdir": "stable-video-diffusion",
        },
        {
            "label": "ControlNet checkpoint",
            "repo_id": "lllyasviel/sd-controlnet-canny",
            "filename": "diffusion_pytorch_model.safetensors",
            "revision": "7f2f69197050967007f6bbd23ab5e52f0384162a",
            "subdir": "controlnet",
        },
    ],
    "base": [
        {
            "label": "Stable Video Diffusion config",
            "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt",
            "filename": "model_index.json",
            "revision": "9e43909513c6714f1bc78bcb44d96e733cd242aa",
            "subdir": "stable-video-diffusion",
        },
        {
            "label": "Stable Video Diffusion UNet",
            "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt",
            "filename": "unet/diffusion_pytorch_model.safetensors",
            "revision": "9e43909513c6714f1bc78bcb44d96e733cd242aa",
            "subdir": "stable-video-diffusion",
        },
        {
            "label": "ControlNet checkpoint",
            "repo_id": "lllyasviel/sd-controlnet-canny",
            "filename": "diffusion_pytorch_model.safetensors",
            "revision": "7f2f69197050967007f6bbd23ab5e52f0384162a",
            "subdir": "controlnet",
        },
    ],
    "large": [
        {
            "label": "Stable Video Diffusion config",
            "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
            "filename": "model_index.json",
            "revision": "043843887ccd51926e3efed36270444a838e7861",
            "subdir": "stable-video-diffusion",
        },
        {
            "label": "Stable Video Diffusion UNet",
            "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
            "filename": "unet/diffusion_pytorch_model.safetensors",
            "revision": "043843887ccd51926e3efed36270444a838e7861",
            "subdir": "stable-video-diffusion",
        },
        {
            "label": "ControlNet checkpoint",
            "repo_id": "lllyasviel/control_v11p_sd15_canny",
            "filename": "diffusion_pytorch_model.safetensors",
            "revision": "115a470d547982438f70198e353a921996e2e819",
            "subdir": "controlnet",
        },
    ],
}


def _target_path(asset: dict[str, str], output_dir: Path) -> Path:
    return output_dir / asset["subdir"] / Path(asset["filename"])


def _download_asset(asset: dict[str, str], output_dir: Path, dry_run: bool) -> tuple[str, str]:
    target_path = _target_path(asset, output_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        return asset["label"], f"skipped (already exists at {target_path})"

    if dry_run:
        return asset["label"], f"would download {asset['repo_id']}:{asset['filename']} -> {target_path}"

    local_dir = str(output_dir / asset["subdir"])
    local_path = hf_hub_download(
        repo_id=asset["repo_id"],
        filename=asset["filename"],
        revision=asset["revision"],
        local_dir=local_dir,
    )
    return asset["label"], f"downloaded to {local_path}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pretrained weights for cinematic-controlnet")
    parser.add_argument("--output-dir", default="checkpoints", help="Directory to store downloaded assets")
    parser.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default="base")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without fetching files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing '{args.model_size}' preset in {output_dir}")
    for asset in MODEL_PRESETS[args.model_size]:
        label, status = _download_asset(asset, output_dir, args.dry_run)
        print(f"- {label}: {status}")

    print("- Adapter weights: random init fallback remains in place for repo-specific layers.")


if __name__ == "__main__":
    main()
