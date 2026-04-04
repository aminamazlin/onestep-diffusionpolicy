"""
Thin wrapper around the upstream DiffusionUnetImagePolicy from the
open-source diffusion_policy library.

We re-export it here so the rest of the OneDP codebase only needs to
import from `onedp.models` rather than knowing the upstream package layout.
The wrapper also adds a `save_checkpoint` / `load_checkpoint` convenience
and a `encode_obs` helper used by the distillation training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict

# ── Upstream library imports ──────────────────────────────────────────────────
# diffusion_policy must be on PYTHONPATH (it is a source repo, not a pip package):
#   git clone https://github.com/real-stanford/diffusion_policy.git
#   export PYTHONPATH=$PYTHONPATH:/path/to/diffusion_policy
try:
    from diffusion_policy.policy.diffusion_unet_image_policy import (
        DiffusionUnetImagePolicy as _UpstreamPolicy,
    )
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    from diffusion_policy.model.vision.multi_image_obs_encoder import (
        MultiImageObsEncoder,
    )
except ImportError as e:
    raise ImportError(
        "diffusion_policy not found on PYTHONPATH.  Clone the repo and set:\n"
        "  git clone https://github.com/real-stanford/diffusion_policy.git\n"
        "  export PYTHONPATH=$PYTHONPATH:/path/to/diffusion_policy\n"
        f"Original error: {e}"
    )


class DiffusionPolicy(_UpstreamPolicy):
    """
    Upstream DiffusionUnetImagePolicy with added checkpoint helpers and an
    `encode_obs` method that returns raw observation features — needed by
    the OneDP distillation loop to share the same obs encoding between the
    generator and the frozen teacher.

    All constructor arguments are forwarded unchanged to the upstream class.
    See upstream source for full documentation.
    """

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode an observation dictionary into a global conditioning vector.

        The upstream policy does this inside `predict_action` / `compute_loss`
        but does not expose it publicly.  We replicate the same logic here so
        that the distillation loop can share one obs-encoding call for both the
        generator and the frozen teacher score network.

        Returns:
            (B, obs_feature_dim) conditioning vector
        """
        nobs = self.normalizer.normalize(obs_dict)

        if self.obs_as_global_cond:
            # Flatten obs across time: (B, n_obs_steps, feature_dim) → (B, global_cond_dim)
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
            )
            obs_features = self.obs_encoder(this_nobs)   # (B*n_obs, feat)
            B = next(iter(obs_dict.values())).shape[0]
            obs_features = obs_features.reshape(B, -1)   # (B, global_cond_dim)
        else:
            raise NotImplementedError(
                "encode_obs is only implemented for obs_as_global_cond=True"
            )
        return obs_features

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path, epoch: int, **extra):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"epoch": epoch, "state_dict": self.state_dict(), **extra},
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        model: "DiffusionPolicy",
        strict: bool = True,
    ) -> int:
        """Load weights into an already-constructed model.  Returns epoch."""
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=strict)
        return ckpt.get("epoch", 0)


# ── re-export upstream building blocks for convenience ───────────────────────
__all__ = ["DiffusionPolicy", "ConditionalUnet1D", "MultiImageObsEncoder"]


# ── local helper (mirrors upstream _apply_transform_to_dict) ─────────────────
def dict_apply(d: dict, fn) -> dict:
    return {k: fn(v) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
