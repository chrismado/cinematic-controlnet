"""
Training Script: Neural Continuum Physics Solver

Full training loop with:
  - Accelerate for distributed training
  - AdamW optimizer with cosine LR schedule
  - Weights & Biases logging
  - Periodic checkpointing
  - Gradient clipping

Usage:
    accelerate launch training/train_physics_solver.py \
        --epochs 100 --batch_size 16 --lr 1e-4 --output_dir checkpoints/solver
"""
import argparse
import math
import os
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

from physics.neural_continuum_solver import NeuralContinuumSolver


def create_synthetic_dataset(
    num_samples: int = 1000,
    seq_len: int = 64,
    latent_dim: int = 512,
    output_h: int = 60,
    output_w: int = 104,
) -> TensorDataset:
    """Create synthetic training data for development/testing."""
    latent_states = torch.randn(num_samples, seq_len, latent_dim)
    force_vectors = torch.randn(num_samples, 6)
    # Synthetic targets
    target_flow = torch.randn(num_samples, output_h, output_w, 2)
    target_rgb = torch.sigmoid(torch.randn(num_samples, output_h, output_w, 3))
    return TensorDataset(latent_states, force_vectors, target_flow, target_rgb)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine learning rate schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args):
    # Initialize accelerator
    if Accelerator is not None:
        accelerator = Accelerator(log_with="wandb" if args.use_wandb and wandb else None)
        device = accelerator.device
    else:
        accelerator = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    if args.use_wandb and wandb is not None:
        if accelerator is not None:
            accelerator.init_trackers(
                project_name=args.wandb_project,
                config=vars(args),
            )
        else:
            wandb.init(project=args.wandb_project, config=vars(args))

    # Model
    model = NeuralContinuumSolver(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        output_h=args.output_h,
        output_w=args.output_w,
    )

    # Dataset
    dataset = create_synthetic_dataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        latent_dim=args.latent_dim,
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

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LR schedule
    num_training_steps = len(dataloader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Prepare with accelerator
    if accelerator is not None:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
    else:
        model = model.to(device)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (latent, force, target_flow, target_rgb) in enumerate(dataloader):
            if accelerator is None:
                latent = latent.to(device)
                force = force.to(device)
                target_flow = target_flow.to(device)
                target_rgb = target_rgb.to(device)

            # Forward
            pred_flow, pred_rgb = model(latent, force)

            # Loss
            flow_loss = F.mse_loss(pred_flow, target_flow)
            rgb_loss = F.mse_loss(pred_rgb, target_rgb)
            loss = flow_loss + args.rgb_weight * rgb_loss

            # Backward
            if accelerator is not None:
                accelerator.backward(loss)
            else:
                loss.backward()

            # Gradient clipping
            if args.max_grad_norm > 0:
                if accelerator is not None:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            # Logging
            if global_step % args.log_every == 0:
                log_dict = {
                    "loss": loss.item(),
                    "flow_loss": flow_loss.item(),
                    "rgb_loss": rgb_loss.item(),
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
                    f"[Epoch {epoch+1}/{args.epochs}] "
                    f"Step {global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Flow: {flow_loss.item():.4f} | "
                    f"RGB: {rgb_loss.item():.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} complete. Avg loss: {avg_loss:.4f}")

        # Checkpointing
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = output_dir / f"solver_epoch_{epoch+1}.pt"
            state = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict": model.state_dict() if accelerator is None
                    else accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": {
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "output_h": args.output_h,
                    "output_w": args.output_w,
                },
            }
            if accelerator is not None:
                accelerator.save(state, str(ckpt_path))
            else:
                torch.save(state, str(ckpt_path))
            print(f"Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = output_dir / "solver_final.pt"
    state = {
        "epoch": args.epochs,
        "global_step": global_step,
        "state_dict": model.state_dict() if accelerator is None
            else accelerator.unwrap_model(model).state_dict(),
        "config": {
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "output_h": args.output_h,
            "output_w": args.output_w,
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


def main():
    parser = argparse.ArgumentParser(description="Train Neural Continuum Physics Solver")

    # Model
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_h", type=int, default=60)
    parser.add_argument("--output_w", type=int, default=104)

    # Data
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--rgb_weight", type=float, default=0.5)

    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cinematic-controlnet-solver")
    parser.add_argument("--log_every", type=int, default=10)

    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="checkpoints/solver")
    parser.add_argument("--save_every", type=int, default=10)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
