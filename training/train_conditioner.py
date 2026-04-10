"""
Training Script: Flow Conditioner

Trains the FlowConditioner module with a frozen diffusion backbone.
The conditioner learns to translate physics solver outputs (optical flow
+ coarse RGB) into effective conditioning signals for diffusion.

Usage:
    accelerate launch training/train_conditioner.py \
        --epochs 50 --batch_size 8 --lr 5e-5 --output_dir checkpoints/conditioner
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None

try:
    import wandb
except ImportError:
    wandb = None

from conditioning.flow_conditioner import FlowConditioner
from inference.realtime_pipeline import StubDiffusionModel


def create_synthetic_dataset(
    num_samples: int = 500,
    flow_h: int = 60,
    flow_w: int = 104,
    output_h: int = 480,
    output_w: int = 832,
) -> TensorDataset:
    """Create synthetic flow/RGB pairs with target frames."""
    flows = torch.randn(num_samples, flow_h, flow_w, 2)
    rgbs = torch.sigmoid(torch.randn(num_samples, flow_h, flow_w, 3))
    # Target frames (simulating what diffusion should produce)
    targets = torch.sigmoid(torch.randn(num_samples, 3, output_h // 8, output_w // 8))
    return TensorDataset(flows, rgbs, targets)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine LR schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args: argparse.Namespace) -> None:
    # Initialize accelerator
    if Accelerator is not None:
        accelerator = Accelerator(log_with="wandb" if args.use_wandb and wandb else None)
        device = accelerator.device
    else:
        accelerator = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_wandb and wandb is not None:
        if accelerator is not None:
            accelerator.init_trackers(
                project_name=args.wandb_project,
                config=vars(args),
            )
        else:
            wandb.init(project=args.wandb_project, config=vars(args))

    conditioning_dim = 1280 if args.backend == "hunyuan" else 1024

    # Trainable conditioner
    conditioner = FlowConditioner(
        hidden_dim=args.hidden_dim,
        conditioning_dim=conditioning_dim,
        backend=args.backend,
    )

    # Frozen diffusion stub (in production, use actual backbone)
    diffusion = StubDiffusionModel(
        conditioning_dim=conditioning_dim,
        output_h=args.output_h,
        output_w=args.output_w,
    )
    diffusion.eval()
    for param in diffusion.parameters():
        param.requires_grad = False

    # Dataset
    dataset = create_synthetic_dataset(
        num_samples=args.num_samples,
        flow_h=args.flow_h,
        flow_w=args.flow_w,
        output_h=args.output_h,
        output_w=args.output_w,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Optimizer (only conditioner parameters)
    optimizer = torch.optim.AdamW(
        conditioner.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    num_training_steps = len(dataloader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    if accelerator is not None:
        conditioner, optimizer, dataloader, scheduler = accelerator.prepare(
            conditioner, optimizer, dataloader, scheduler
        )
        diffusion = diffusion.to(device)
    else:
        conditioner = conditioner.to(device)
        diffusion = diffusion.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        conditioner.train()
        epoch_loss = 0.0

        for batch_idx, (flow, rgb, target) in enumerate(dataloader):
            if accelerator is None:
                flow = flow.to(device)
                rgb = rgb.to(device)
                target = target.to(device)

            # Forward through conditioner
            conditioning = conditioner(flow, rgb)

            # Forward through frozen diffusion
            with torch.no_grad():
                output = diffusion(conditioning)

            # Downsample output to match target size
            if output.shape[2:] != target.shape[2:]:
                output_ds = F.interpolate(
                    output,
                    size=target.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                output_ds = output

            # Reconstruction loss
            loss = F.mse_loss(output_ds, target)

            if accelerator is not None:
                accelerator.backward(loss)
            else:
                loss.backward()

            if args.max_grad_norm > 0:
                if accelerator is not None:
                    accelerator.clip_grad_norm_(conditioner.parameters(), args.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(conditioner.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                log_dict = {
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step": global_step,
                }
                if args.use_wandb and wandb is not None:
                    if accelerator is not None:
                        accelerator.log(log_dict, step=global_step)
                    else:
                        wandb.log(log_dict, step=global_step)

                print(
                    f"[Epoch {epoch + 1}/{args.epochs}] "
                    f"Step {global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs} complete. Avg loss: {avg_loss:.4f}")

        # Checkpointing
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = output_dir / f"conditioner_epoch_{epoch + 1}.pt"
            state = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict": conditioner.state_dict()
                if accelerator is None
                else accelerator.unwrap_model(conditioner).state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": {
                    "hidden_dim": args.hidden_dim,
                    "conditioning_dim": conditioning_dim,
                    "backend": args.backend,
                },
            }
            if accelerator is not None:
                accelerator.save(state, str(ckpt_path))
            else:
                torch.save(state, str(ckpt_path))
            print(f"Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = output_dir / "conditioner_final.pt"
    state = {
        "epoch": args.epochs,
        "global_step": global_step,
        "state_dict": conditioner.state_dict()
        if accelerator is None
        else accelerator.unwrap_model(conditioner).state_dict(),
        "config": {
            "hidden_dim": args.hidden_dim,
            "conditioning_dim": conditioning_dim,
            "backend": args.backend,
        },
    }
    if accelerator is not None:
        accelerator.save(state, str(final_path))
    else:
        torch.save(state, str(final_path))
    print(f"Training complete. Final model saved to {final_path}")

    if args.use_wandb and wandb is not None:
        if accelerator is not None:
            accelerator.end_training()
        else:
            wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Flow Conditioner")

    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--backend", type=str, default="hunyuan", choices=["hunyuan", "wan2"])

    # Data
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--flow_h", type=int, default=60)
    parser.add_argument("--flow_w", type=int, default=104)
    parser.add_argument("--output_h", type=int, default=480)
    parser.add_argument("--output_w", type=int, default=832)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cinematic-controlnet-conditioner")
    parser.add_argument("--log_every", type=int, default=10)

    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="checkpoints/conditioner")
    parser.add_argument("--save_every", type=int, default=10)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
