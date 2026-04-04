"""
EDM (Elucidating the Design Space of Diffusion Models, Karras et al. 2022)
noise schedule utilities for OneDP distillation.

The paper uses EDM for the second set of experiments (required by the
Consistency Policy baseline).  Key differences vs DDPM:
  - Continuous-time parameterisation: noise level σ ∈ [σ_min, σ_max]
  - Noise sampled from log-normal: log σ ~ N(P_mean, P_std²)
  - Preconditioning: the network F_θ is scaled so training is stable
      D_θ(x; σ, c) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x, c_noise(σ), c)
  - Second-order (Heun) ODE sampler during inference

For distillation we:
  - sample log σ ~ N(P_mean, P_std²), clamp to [σ_min, σ_max]
  - forward-diffuse:  x_σ = x_0 + σ * ε,  ε ~ N(0, I)
  - compute score:  s(x_σ) = (x_0_pred - x_σ) / σ²  = -ε_eff / σ
  - apply score-difference KL gradient

Generator fixed σ:  σ_init = 2.5  (paper Appendix B)
"""

import math
import torch
from typing import Tuple


class EDMDistillationScheduler:
    """
    EDM noise schedule utilities for OneDP distillation.

    Args:
        sigma_min:  minimum noise level  (EDM default 0.002)
        sigma_max:  maximum noise level  (EDM default 80.0)
        sigma_data: expected data std    (EDM default 0.5)
        P_mean:     log-normal mean for noise sampling (EDM default -1.2)
        P_std:      log-normal std  for noise sampling (EDM default 1.2)
        sigma_init: fixed σ fed to the generator at inference/init (paper: 2.5)
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_init: float = 2.5,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_init = sigma_init

    # ------------------------------------------------------------------
    # EDM preconditioning coefficients  (Table 1 in Karras et al. 2022)
    # ------------------------------------------------------------------

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1.0 / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        """Map σ to the noise embedding input (log-scaled)."""
        return 0.25 * sigma.log()

    # ------------------------------------------------------------------
    # Training loss weight  λ(σ)  (EDM Eq. 5 / paper Appendix)
    # ------------------------------------------------------------------

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """λ(σ) = (σ² + σ_data²) / (σ · σ_data)²"""
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

    # ------------------------------------------------------------------
    # Noise sampling for training / distillation
    # ------------------------------------------------------------------

    def sample_sigmas(self, batch_size: int, device) -> torch.Tensor:
        """
        Sample σ from log-normal distribution used during EDM training.
        log σ ~ N(P_mean, P_std²), clamped to [σ_min, σ_max].
        """
        log_sigma = torch.randn(batch_size, device=device) * self.P_std + self.P_mean
        sigma = log_sigma.exp().clamp(self.sigma_min, self.sigma_max)
        return sigma

    # ------------------------------------------------------------------
    # Forward diffusion   x_σ = x_0 + σ * ε
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        sigma: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffuse x_0 to noise level σ.

        Args:
            x0:    (B, T_pred, action_dim)
            sigma: (B,)  noise levels
            noise: optional (B, T_pred, action_dim)
        Returns:
            (x_sigma, noise)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sigma_bc = sigma.view(-1, 1, 1).to(x0)
        x_sigma = x0 + sigma_bc * noise
        return x_sigma, noise

    # ------------------------------------------------------------------
    # Preconditioned network call   D_θ(x; σ, c)
    # ------------------------------------------------------------------

    def precondition_input(
        self, x_sigma: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Scale the noisy input:  c_in(σ) · x_σ"""
        return self.c_in(sigma).view(-1, 1, 1).to(x_sigma) * x_sigma

    def precondition_output(
        self,
        x_sigma: torch.Tensor,
        sigma: torch.Tensor,
        F_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine skip connection and network output:
          D = c_skip(σ) · x_σ  +  c_out(σ) · F_out
        Returns the denoised x_0 estimate.
        """
        c_s = self.c_skip(sigma).view(-1, 1, 1).to(x_sigma)
        c_o = self.c_out(sigma).view(-1, 1, 1).to(x_sigma)
        return c_s * x_sigma + c_o * F_out

    # ------------------------------------------------------------------
    # Score computation   s(x_σ) = (D(x_σ) - x_σ) / σ²
    # ------------------------------------------------------------------

    def score_from_denoised(
        self, x_sigma: torch.Tensor, x0_pred: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Compute score from denoised estimate: (x_0_pred - x_σ) / σ²."""
        sigma_bc = sigma.view(-1, 1, 1).to(x_sigma)
        return (x0_pred - x_sigma) / sigma_bc ** 2

    # ------------------------------------------------------------------
    # Distillation weight   w(σ) = σ²  (analogous to DDPM σ²)
    # ------------------------------------------------------------------

    def distillation_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        KL weighting for EDM distillation.
        Returns σ (= sqrt(w(σ))) shaped (B, 1, 1).
        """
        return sigma.view(-1, 1, 1)

    # ------------------------------------------------------------------
    # Noise level for generator fixed-σ  (σ_init = 2.5)
    # ------------------------------------------------------------------

    def generator_sigma(self, batch_size: int, device) -> torch.Tensor:
        """Return the fixed σ used as generator input."""
        return torch.full((batch_size,), self.sigma_init, device=device)
