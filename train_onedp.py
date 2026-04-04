"""
Distil a pre-trained Diffusion Policy into a One-Step Diffusion Policy (OneDP).

Implements Algorithm 1 from Wang et al. 2024 (arXiv:2410.21257).

Two variants:
  --variant stochastic   OneDP-S: generator + score network  (Eq. 5 + 6)
  --variant deterministic OneDP-D: generator only            (Eq. 8)

Two noise schedules:
  --schedule ddpm   discrete-time, 100 steps   (pre-trained with DDPM)
  --schedule edm    continuous-time EDM         (pre-trained with EDM)

Usage (simulation, DDPM):
  python train_onedp.py \
      --pretrained_ckpt outputs/dp_square_ph_ddpm/dp_final.ckpt \
      --dataset_path    data/robomimic/square_ph.hdf5 \
      --output_dir      outputs/onedp_square_ph_ddpm \
      --schedule ddpm   --variant stochastic \
      --num_epochs 20

Usage (real-world, DDPM):
  python train_onedp.py \
      --pretrained_ckpt outputs/dp_pnp_milk_ddpm/dp_final.ckpt \
      --dataset_path    data/real/pnp_milk.hdf5 \
      --output_dir      outputs/onedp_pnp_milk \
      --schedule ddpm   --variant stochastic \
      --num_epochs 100

Hyperparameters (from paper Appendix B / Table 7):
  generator LR        = 1e-6
  score network LR    = 2e-5
  optimizer β₁        = 0.0   (both nets, GAN-style)
  optimizer β₂        = 0.999
  action chunk size   = 16
  n_obs_steps         = 2
  DDPM distill t range = [2, 95]
  DDPM generator t_init = 65
  EDM generator σ_init  = 2.5
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from onedp.data.dataset import RobomimicDataset
from onedp.models.onedp import OneDP
from onedp.schedulers.ddpm import DDPMDistillationScheduler
from onedp.schedulers.edm import EDMDistillationScheduler


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Train OneDP (distillation)")

    # Paths
    p.add_argument("--pretrained_ckpt", required=True,
                   help="Path to pre-trained DiffusionPolicy checkpoint (.ckpt)")
    p.add_argument("--dataset_path", required=True,
                   help="Path to Robomimic HDF5 dataset")
    p.add_argument("--output_dir", required=True,
                   help="Directory for checkpoints and logs")

    # Algorithm
    p.add_argument("--variant", choices=["stochastic", "deterministic"],
                   default="stochastic",
                   help="OneDP-S (stochastic) or OneDP-D (deterministic)")
    p.add_argument("--schedule", choices=["ddpm", "edm"],
                   default="ddpm",
                   help="Noise schedule used when pre-training the policy")

    # Training
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr_generator", type=float, default=1e-6)
    p.add_argument("--lr_score_net", type=float, default=2e-5)
    p.add_argument("--device", default="cuda")

    # Architecture (must match pre-trained policy)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--n_obs_steps", type=int, default=2)
    p.add_argument("--n_action_steps", type=int, default=8)
    p.add_argument("--t_init", type=int, default=65,
                   help="Fixed timestep for generator input (DDPM). Paper: 65")

    # Logging
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--use_wandb", action="store_true")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────────
    if args.use_wandb:
        import wandb
        wandb.init(project="onedp", config=vars(args))

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = RobomimicDataset(
        dataset_path=args.dataset_path,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    action_shape = (args.horizon, dataset.action_dim)
    print(f"Dataset: {len(dataset)} samples  |  action_shape={action_shape}")

    # ── Load pre-trained policy ───────────────────────────────────────────────
    print(f"Loading pre-trained policy from {args.pretrained_ckpt} ...")
    pretrained_policy = _load_pretrained_policy(
        args.pretrained_ckpt, dataset, device
    )
    pretrained_noise_pred = pretrained_policy.model  # ConditionalUnet1D

    # ── Noise schedule ────────────────────────────────────────────────────────
    if args.schedule == "ddpm":
        scheduler = DDPMDistillationScheduler().to(device)
    else:
        scheduler = EDMDistillationScheduler()

    # ── Build OneDP model ─────────────────────────────────────────────────────
    print(f"Building OneDP-{'S' if args.variant == 'stochastic' else 'D'} ...")
    onedp = OneDP.from_pretrained(
        pretrained_noise_pred=pretrained_noise_pred,
        scheduler=scheduler,
        variant=args.variant,
        t_init=args.t_init,
    ).to(device)

    # ── Optimisers ────────────────────────────────────────────────────────────
    # Paper uses Adam with β₁=0, β₂=0.999 for both nets (GAN-style).
    gen_optimizer = torch.optim.Adam(
        onedp.generator.parameters(),
        lr=args.lr_generator,
        betas=(0.0, 0.999),
    )
    optimizers = [gen_optimizer]

    score_optimizer = None
    if args.variant == "stochastic" and onedp.score_network is not None:
        score_optimizer = torch.optim.Adam(
            onedp.score_network.parameters(),
            lr=args.lr_score_net,
            betas=(0.0, 0.999),
        )
        optimizers.append(score_optimizer)

    # ── Observation encoder from pre-trained policy (frozen) ─────────────────
    # The obs encoder is shared: we freeze it during distillation.
    obs_encoder = pretrained_policy.obs_encoder.to(device)
    for p in obs_encoder.parameters():
        p.requires_grad_(False)
    obs_encoder.eval()

    # ── Training loop ─────────────────────────────────────────────────────────
    normalizer = dataset.get_normalizer().to(device)

    for epoch in range(1, args.num_epochs + 1):
        onedp.generator.train()
        if onedp.score_network is not None:
            onedp.score_network.train()

        epoch_stats = {"loss_gen": 0.0, "loss_score_net": 0.0}

        for batch in loader:
            batch = _move_to_device(batch, device)

            # ── Encode observations ───────────────────────────────────────
            with torch.no_grad():
                obs_features = _encode_obs(obs_encoder, batch["obs"], normalizer)
                # obs_features: (B, obs_cond_dim)

            # ── Compute distillation losses ───────────────────────────────
            losses = onedp.compute_loss(obs_features, action_shape)

            # ── Update score network first (before generator) ─────────────
            # This ensures π_ψ tracks the generator's distribution before
            # the generator updates — analogous to the critic/discriminator
            # update in GAN training.
            if score_optimizer is not None:
                score_optimizer.zero_grad()
                losses["loss_score_net"].backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(
                    onedp.score_network.parameters(), 10.0
                )
                score_optimizer.step()

            # ── Update generator ──────────────────────────────────────────
            gen_optimizer.zero_grad()
            losses["loss_generator"].backward()
            torch.nn.utils.clip_grad_norm_(
                onedp.generator.parameters(), 10.0
            )
            gen_optimizer.step()

            epoch_stats["loss_gen"] += losses["loss_generator"].item()
            epoch_stats["loss_score_net"] += losses["loss_score_net"].item()

        # ── Logging ───────────────────────────────────────────────────────
        n = len(loader)
        avg_gen   = epoch_stats["loss_gen"] / n
        avg_score = epoch_stats["loss_score_net"] / n

        if epoch % args.log_every == 0:
            print(
                f"Epoch {epoch:3d}/{args.num_epochs}"
                f"  loss_gen={avg_gen:.4e}"
                f"  loss_score={avg_score:.4e}"
            )
            if args.use_wandb:
                import wandb
                wandb.log({"epoch": epoch,
                           "loss_gen": avg_gen,
                           "loss_score_net": avg_score})

        if epoch % args.save_every == 0 or epoch == args.num_epochs:
            ckpt_path = output_dir / f"onedp_epoch_{epoch:04d}.ckpt"
            onedp.save_checkpoint(ckpt_path, epoch=epoch)
            print(f"  → saved {ckpt_path}")

    print(f"\nDistillation complete.  Outputs in {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_pretrained_policy(ckpt_path: str, dataset: RobomimicDataset, device):
    """
    Load a DiffusionUnetImagePolicy from a checkpoint produced by train_dp.py.
    The policy is returned in eval mode with frozen weights.
    """
    try:
        from diffusion_policy.policy.diffusion_unet_image_policy import (
            DiffusionUnetImagePolicy,
        )
    except ImportError:
        sys.exit(
            "diffusion_policy not found on PYTHONPATH.\n"
            "  git clone https://github.com/real-stanford/diffusion_policy.git\n"
            "  export PYTHONPATH=$PYTHONPATH:/path/to/diffusion_policy"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Re-build the policy with the same architecture, then load weights.
    # In production this is done via Hydra; here we call our builder.
    import train_dp as dp_mod
    policy = dp_mod._build_policy(
        _FakeCfg(dataset),
        dataset,
        device,
    )
    policy.load_state_dict(ckpt["state_dict"])
    policy.set_normalizer(dataset.get_normalizer())
    policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


def _encode_obs(obs_encoder, obs_dict: Dict, normalizer) -> torch.Tensor:
    """
    Encode observation dict → global conditioning vector.

    Normalises low-dim observations using the public normalizer API, then
    passes the full dict through the obs encoder to produce a flat feature
    vector per sample.  Image tensors (already in [0, 1]) are passed through
    unchanged since no image normalisation stats are fitted.
    """
    normed = {}
    for key, val in obs_dict.items():
        if key in normalizer.stats:
            # Low-dim key with fitted stats → normalise to [-1, 1]
            normed[key] = normalizer.normalize({key: val})[key]
        else:
            # Image key (or unfitted key) → pass through unchanged
            normed[key] = val

    # Flatten obs-timestep axis: (B, n_obs, ...) → (B*n_obs, ...)
    B = next(iter(normed.values())).shape[0]
    flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in normed.items()}

    features = obs_encoder(flat)    # (B*n_obs, feat_dim)
    features = features.reshape(B, -1)  # (B, n_obs * feat_dim)
    return features


def _move_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch


class _FakeCfg:
    """Minimal config object to pass to _build_policy.

    Uses the dataset's actual image/lowdim keys so the rebuilt policy
    architecture exactly matches the saved checkpoint.
    """
    def __init__(self, dataset: RobomimicDataset):
        self.horizon = 16
        self.n_obs_steps = 2
        self.n_action_steps = 8
        self.num_train_timesteps = 100
        self.num_inference_steps = 100
        self.obs_as_global_cond = True
        # Mirror the dataset's observation keys to ensure shape compatibility
        self._image_keys = list(dataset.image_keys)
        self._lowdim_keys = list(dataset.lowdim_keys)

    def get(self, key, default=None):
        if key == "image_keys":
            return self._image_keys
        if key == "lowdim_keys":
            return self._lowdim_keys
        return getattr(self, key, default)


if __name__ == "__main__":
    main()
