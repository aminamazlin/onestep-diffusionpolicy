"""
Robomimic HDF5 dataset loader.

Supports the benchmark tasks used in the paper:
  PushT, Square-mh/ph, ToolHang-ph, Transport-mh/ph

Each HDF5 file has the structure produced by robomimic's data collection:
  data/
    demo_0/
      obs/
        agentview_image:    (T, H, W, 3)  uint8
        robot0_eye_in_hand: (T, H, W, 3)  uint8  [optional wrist cam]
        robot0_eef_pos:     (T, 3)
        robot0_eef_quat:    (T, 4)
        robot0_gripper_qpos:(T, 2)
      actions:              (T, action_dim)
    demo_1/
      ...

The dataset returns overlapping windows of `horizon` timesteps.
The first `n_obs_steps` steps of each window are observations;
the last `n_action_steps` steps are the action chunk to predict.

Compatible with the upstream diffusion_policy DataLoader convention:
  batch["obs"]["agentview_image"]  → (B, n_obs_steps, H, W, 3)
  batch["action"]                  → (B, horizon, action_dim)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from onedp.data.normalizer import LinearNormalizer


class RobomimicDataset(Dataset):
    """
    Sliding-window dataset over Robomimic HDF5 demonstrations.

    Args:
        dataset_path:   path to the .hdf5 file
        horizon:        total window length  (obs + action chunk)
        n_obs_steps:    number of observation steps at the start of the window
        n_action_steps: number of action steps to predict
                        (n_obs_steps + n_action_steps must equal horizon)
        image_keys:     which image observation keys to include
        lowdim_keys:    which low-dimensional observation keys to include
        pad_before:     replicate first frame to handle episode boundaries
        pad_after:      replicate last  frame to handle episode boundaries
        normalizer:     pre-fitted LinearNormalizer; fitted from data if None
    """

    def __init__(
        self,
        dataset_path: str | Path,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        image_keys: Tuple[str, ...] = (
            "agentview_image",
            "robot0_eye_in_hand_image",
        ),
        lowdim_keys: Tuple[str, ...] = (
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ),
        pad_before: int = 0,
        pad_after: int = 0,
        normalizer: Optional[LinearNormalizer] = None,
    ):
        assert n_obs_steps + n_action_steps <= horizon, (
            "n_obs_steps + n_action_steps must be <= horizon"
        )
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.image_keys = list(image_keys)
        self.lowdim_keys = list(lowdim_keys)
        self.pad_before = pad_before
        self.pad_after = pad_after

        # ── Load all demos into memory ────────────────────────────────────
        dataset_path = Path(dataset_path)
        assert dataset_path.exists(), f"Dataset not found: {dataset_path}"

        self.episodes: List[Dict[str, np.ndarray]] = []
        self._load_hdf5(dataset_path)

        # ── Build (episode_idx, start_t) index ──────────────────────────
        self._index: List[Tuple[int, int]] = []
        for ep_idx, ep in enumerate(self.episodes):
            T = len(ep["action"])
            T_padded = T + pad_before + pad_after
            for t in range(T_padded - horizon + 1):
                self._index.append((ep_idx, t - pad_before))

        # ── Fit normaliser if not provided ───────────────────────────────
        if normalizer is None:
            normalizer = self._fit_normalizer()
        self.normalizer = normalizer

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_hdf5(self, path: Path):
        with h5py.File(path, "r") as f:
            demo_keys = sorted(f["data"].keys())
            for dk in demo_keys:
                demo = f["data"][dk]
                ep: Dict[str, np.ndarray] = {}

                # Actions
                ep["action"] = demo["actions"][()].astype(np.float32)

                # Image observations
                for key in self.image_keys:
                    full_key = f"obs/{key}"
                    if full_key in demo:
                        ep[key] = demo[full_key][()]  # (T, H, W, C) uint8
                    # silently skip missing cameras

                # Low-dim observations
                for key in self.lowdim_keys:
                    full_key = f"obs/{key}"
                    if full_key in demo:
                        ep[key] = demo[full_key][()].astype(np.float32)

                self.episodes.append(ep)

    # ------------------------------------------------------------------
    # Normaliser fitting (from raw actions)
    # ------------------------------------------------------------------

    def _fit_normalizer(self) -> LinearNormalizer:
        all_actions = np.concatenate(
            [ep["action"] for ep in self.episodes], axis=0
        )
        norm = LinearNormalizer()
        norm.fit({"action": all_actions})

        # Fit low-dim obs normalisation too (same [-1, 1] range as actions)
        for key in self.lowdim_keys:
            values = [ep[key] for ep in self.episodes if key in ep]
            if values:
                all_vals = np.concatenate(values, axis=0)
                norm.fit({key: all_vals})

        return norm

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict:
        ep_idx, start_t = self._index[idx]
        ep = self.episodes[ep_idx]
        T = len(ep["action"])

        def clamp(t):
            return max(0, min(t, T - 1))

        # Build window indices, clamping at boundaries (pad by replication)
        ts = [clamp(start_t + i) for i in range(self.horizon)]

        # ── Actions ──────────────────────────────────────────────────────
        action_seq = ep["action"][ts]  # (H, action_dim)
        action_tensor = torch.from_numpy(action_seq)
        normed_action = self.normalizer.normalize_action(action_tensor)

        # ── Observations (first n_obs_steps timesteps of window) ─────────
        obs_ts = ts[: self.n_obs_steps]
        obs: Dict[str, torch.Tensor] = {}

        for key in self.image_keys:
            if key in ep:
                imgs = ep[key][obs_ts]               # (n_obs, H, W, C)
                imgs = torch.from_numpy(imgs)
                # → (n_obs, C, H, W), float in [0, 1]
                imgs = imgs.permute(0, 3, 1, 2).float() / 255.0
                obs[key] = imgs

        for key in self.lowdim_keys:
            if key in ep:
                vals = ep[key][obs_ts]               # (n_obs, dim)
                obs[key] = torch.from_numpy(vals)

        return {
            "obs": obs,                              # dict of (n_obs, ...)
            "action": normed_action,                 # (H, action_dim)
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_normalizer(self) -> LinearNormalizer:
        return self.normalizer

    @property
    def action_dim(self) -> int:
        return self.episodes[0]["action"].shape[-1]

    @property
    def obs_image_shape(self) -> Optional[Tuple[int, ...]]:
        """Returns (C, H, W) of the first image key, or None."""
        for key in self.image_keys:
            if key in self.episodes[0]:
                h, w, c = self.episodes[0][key].shape[1:]
                return (c, h, w)
        return None
