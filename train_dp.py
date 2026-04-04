"""
Pre-train a Diffusion Policy (DP) on Robomimic / PushT data.

This script is a lightweight wrapper that delegates to the upstream
diffusion_policy source code (clone from real-stanford/diffusion_policy and
add to PYTHONPATH — it is not a pip package).
Keeping pre-training separate from distillation makes checkpoints reusable
and matches the paper's two-stage pipeline:
  Stage 1 (this script): train DP for 1000 epochs  → save checkpoint
  Stage 2 (train_onedp.py): distil into OneDP for 20/100 epochs

Usage (DDPM, simulation):
  python train_dp.py \
      --config configs/train_dp_ddpm.yaml \
      dataset_path=data/robomimic/square_ph.hdf5 \
      output_dir=outputs/dp_square_ph_ddpm

Usage (EDM, simulation):
  python train_dp.py \
      --config configs/train_dp_edm.yaml \
      dataset_path=data/robomimic/square_ph.hdf5 \
      output_dir=outputs/dp_square_ph_edm

All Hydra overrides are forwarded to the upstream workspace.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from onedp.data.dataset import RobomimicDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train Diffusion Policy")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config (configs/train_dp_ddpm.yaml or _edm.yaml)"
    )
    # Remaining args are treated as Hydra-style key=value overrides
    args, overrides = parser.parse_known_args()
    return args, overrides


def main():
    args, overrides = parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    for ov in overrides:
        key, val = ov.split("=", 1)
        OmegaConf.update(cfg, key, val)

    print(OmegaConf.to_yaml(cfg))

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_dataset = RobomimicDataset(
        dataset_path=cfg.dataset_path,
        horizon=cfg.horizon,
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        image_keys=tuple(cfg.get("image_keys", ["agentview_image"])),
        lowdim_keys=tuple(cfg.get("lowdim_keys", [])),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    # ── Build model via upstream diffusion_policy ─────────────────────────────
    # The upstream library provides a complete training workspace.
    # We instantiate its DiffusionUnetImagePolicy directly.
    try:
        from diffusion_policy.policy.diffusion_unet_image_policy import (
            DiffusionUnetImagePolicy,
        )
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from diffusion_policy.model.vision.multi_image_obs_encoder import (
            MultiImageObsEncoder,
        )
    except ImportError:
        sys.exit(
            "diffusion_policy not found on PYTHONPATH.\n"
            "  git clone https://github.com/real-stanford/diffusion_policy.git\n"
            "  export PYTHONPATH=$PYTHONPATH:/path/to/diffusion_policy"
        )

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Build model from config (mirrors the upstream workspace approach)
    policy = _build_policy(cfg, train_dataset, device)
    policy.to(device)
    policy.set_normalizer(train_dataset.get_normalizer())

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-6)),
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg.num_epochs)
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(1, int(cfg.num_epochs) + 1):
        policy.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch = _move_to_device(batch, device)
            loss = policy.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()
            epoch_loss += loss.item()

        lr_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        if epoch % cfg.get("log_every", 10) == 0:
            print(f"Epoch {epoch:4d}/{cfg.num_epochs}  loss={avg_loss:.4f}")

        if epoch % cfg.get("save_every", 100) == 0:
            ckpt_path = output_dir / f"dp_epoch_{epoch:04d}.ckpt"
            torch.save(
                {"epoch": epoch, "state_dict": policy.state_dict(),
                 "optimizer": optimizer.state_dict()},
                ckpt_path,
            )
            print(f"  → saved {ckpt_path}")

    # Save final checkpoint
    final_path = output_dir / "dp_final.ckpt"
    torch.save({"epoch": cfg.num_epochs, "state_dict": policy.state_dict()}, final_path)
    print(f"Training complete.  Final checkpoint: {final_path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_policy(cfg, dataset: RobomimicDataset, device):
    """
    Construct a DiffusionUnetImagePolicy from the config.

    Supports two noise schedules (set via cfg.noise_schedule):
      - "ddpm"  (default): DDPM linear schedule, epsilon-prediction, 100 steps
      - "edm":             Karras 2022 schedule, x₀-prediction, Heun sampler

    In production, this is typically done via Hydra instantiation
    (hydra.utils.instantiate).  Here we build it programmatically so that
    train_dp.py has no Hydra dependency at the cost of some boilerplate.
    """
    from diffusion_policy.policy.diffusion_unet_image_policy import (
        DiffusionUnetImagePolicy,
    )

    noise_schedule = cfg.get("noise_schedule", "ddpm")

    if noise_schedule == "edm":
        noise_scheduler = _build_edm_scheduler(cfg)
        prediction_type = "sample"      # EDM uses x₀-prediction
        num_inference_steps = int(cfg.get("num_inference_steps", 18))
    else:
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(cfg.get("num_train_timesteps", 100)),
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        prediction_type = "epsilon"
        num_inference_steps = int(cfg.get("num_inference_steps", 100))

    policy = DiffusionUnetImagePolicy(
        shape_meta=_build_shape_meta(cfg, dataset),
        noise_scheduler=noise_scheduler,
        obs_encoder=_build_obs_encoder(cfg, dataset),
        horizon=int(cfg.horizon),
        n_action_steps=int(cfg.n_action_steps),
        n_obs_steps=int(cfg.n_obs_steps),
        num_inference_steps=num_inference_steps,
        obs_as_global_cond=cfg.get("obs_as_global_cond", True),
    )
    return policy


def _build_edm_scheduler(cfg):
    """
    Build an EDM-compatible noise scheduler.

    Prefers diffusers' EDMEulerScheduler (available in diffusers ≥ 0.21).
    Falls back to a DDPM scheduler configured to approximate the EDM
    σ range via a cosine schedule — sufficient for pre-training but not
    for the exact Heun ODE sampler used in inference.
    """
    sigma_min = float(cfg.get("sigma_min", 0.002))
    sigma_max = float(cfg.get("sigma_max", 80.0))
    sigma_data = float(cfg.get("sigma_data", 0.5))
    num_train_timesteps = int(cfg.get("num_train_timesteps", 40))

    try:
        # diffusers ≥ 0.21 ships EDMEulerScheduler
        from diffusers import EDMEulerScheduler
        return EDMEulerScheduler(
            num_train_timesteps=num_train_timesteps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            prediction_type="sample",
        )
    except ImportError:
        pass

    try:
        # Older diffusers: KarrasVeScheduler approximates the EDM σ range
        from diffusers import KarrasVeScheduler
        return KarrasVeScheduler(
            num_train_timesteps=num_train_timesteps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
    except ImportError:
        pass

    # Last resort: DDPM with squaredcos schedule as a training proxy.
    # NOTE: this does NOT implement the full EDM preconditioning —
    # use a diffusers version that includes EDMEulerScheduler for exact results.
    import warnings
    warnings.warn(
        "EDM-compatible scheduler not found in your diffusers installation. "
        "Falling back to DDPMScheduler with squaredcos schedule. "
        "Upgrade diffusers (pip install -U diffusers) for exact EDM support.",
        UserWarning,
    )
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,
        prediction_type="sample",
    )


def _build_shape_meta(cfg, dataset: RobomimicDataset) -> dict:
    action_dim = dataset.action_dim
    image_shape = dataset.obs_image_shape  # (C, H, W) or None
    meta = {
        "action": {"shape": [action_dim]},
        "obs": {},
    }
    for key in dataset.image_keys:
        if image_shape is not None:
            meta["obs"][key] = {"shape": list(image_shape), "type": "rgb"}
    for key in dataset.lowdim_keys:
        if key in dataset.episodes[0]:
            dim = dataset.episodes[0][key].shape[-1]
            meta["obs"][key] = {"shape": [dim], "type": "low_dim"}
    return meta


def _build_obs_encoder(cfg, dataset: RobomimicDataset):
    from diffusion_policy.model.vision.multi_image_obs_encoder import (
        MultiImageObsEncoder,
    )
    import torchvision.models as tvm

    shape_meta = _build_shape_meta(cfg, dataset)
    rgb_model = tvm.resnet18(weights=None)

    return MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model,
        resize_shape=cfg.get("resize_shape", None),
        crop_shape=cfg.get("crop_shape", None),
        random_crop=cfg.get("random_crop", True),
        use_group_norm=True,
        share_rgb_model=False,
        imagenet_norm=False,
    )


def _move_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


if __name__ == "__main__":
    main()
