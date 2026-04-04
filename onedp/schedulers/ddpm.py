"""
DDPM noise schedule utilities for OneDP distillation.

The paper uses a standard DDPM linear schedule (100 steps) for the
pre-trained diffusion policy.  During distillation we:
  - sample diffusion timestep k ~ Uniform[t_min, t_max]   (default [2, 95])
  - forward-diffuse the generated action at that timestep
  - apply the score-difference KL gradient

Notation follows the paper:
  x^k = α_k * x^0 + σ_k * ε,   ε ~ N(0, I)

where  α_k = sqrt(ᾱ_k)  and  σ_k = sqrt(1 - ᾱ_k)
with   ᾱ_k = ∏_{i=1}^{k} (1 - β_i).

Weighting:  w(k) = σ_k²  (DreamFusion / Eq. 5 in paper)
→ score-diff weight = w(k) / σ_k = σ_k
"""

import torch
import torch.nn as nn
from typing import Tuple


class DDPMDistillationScheduler:
    """
    Wraps a DDPM linear noise schedule and exposes helpers needed for
    OneDP KL distillation (forward diffusion, timestep sampling, weights).

    Args:
        num_train_timesteps: total diffusion steps T (default 100, as in paper)
        beta_start:          β_1 (default 1e-4)
        beta_end:            β_T (default 0.02)
        t_min:               lower bound of distillation timestep range (default 2)
        t_max:               upper bound of distillation timestep range (default 95)
    """

    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        t_min: int = 2,
        t_max: int = 95,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.t_min = t_min
        self.t_max = t_max

        # ── Linear β schedule ───────────────────────────────────────────────
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)  # (T,)
        alphas = 1.0 - betas                                                # (T,)
        alphas_cumprod = torch.cumprod(alphas, dim=0)                       # ᾱ_k (T,)

        # Paper notation: α_k = sqrt(ᾱ_k),  σ_k = sqrt(1 - ᾱ_k)
        self.register("alpha", alphas_cumprod.sqrt())          # (T,)
        self.register("sigma", (1.0 - alphas_cumprod).sqrt())  # (T,)
        self.register("alphas_cumprod", alphas_cumprod)        # (T,)

    # ------------------------------------------------------------------
    # Internal helper: store tensors as plain attributes (no nn.Module)
    # ------------------------------------------------------------------
    def register(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)

    def to(self, device):
        self.alpha = self.alpha.to(device)
        self.sigma = self.sigma.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_timesteps(self, batch_size: int, device) -> torch.Tensor:
        """Sample k ~ Uniform[t_min, t_max] for a batch."""
        return torch.randint(
            self.t_min, self.t_max + 1, (batch_size,), device=device
        )

    def q_sample(
        self,
        x0: torch.Tensor,
        k: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward-diffuse a clean sample to timestep k.

        Returns (x_k, noise) where  x_k = α_k * x_0 + σ_k * noise.

        Args:
            x0:    (B, T_pred, action_dim)  clean action chunk
            k:     (B,)  diffusion timestep indices
            noise: optional (B, T_pred, action_dim); sampled if None
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Broadcast schedule coefficients over action dimensions
        alpha_k = self.alpha[k].view(-1, 1, 1).to(x0)  # (B, 1, 1)
        sigma_k = self.sigma[k].view(-1, 1, 1).to(x0)  # (B, 1, 1)

        x_k = alpha_k * x0 + sigma_k * noise
        return x_k, noise

    def distillation_weight(self, k: torch.Tensor) -> torch.Tensor:
        """
        KL loss weight w(k) = σ_k²  (DreamFusion weighting).

        The score-difference gradient is scaled by w(k)/σ_k = σ_k.
        Returns σ_k shaped (B, 1, 1) ready for broadcasting.
        """
        return self.sigma[k].view(-1, 1, 1)  # = sqrt(w(k)) = σ_k

    def predict_x0_from_noise(
        self,
        x_k: torch.Tensor,
        k: torch.Tensor,
        pred_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recover x_0 estimate from a noise prediction at timestep k.
            x_0 = (x_k - σ_k * ε_pred) / α_k
        Used to convert the generator's ε-prediction into a clean action.
        """
        alpha_k = self.alpha[k].view(-1, 1, 1).to(x_k)
        sigma_k = self.sigma[k].view(-1, 1, 1).to(x_k)
        return (x_k - sigma_k * pred_noise) / alpha_k
