from __future__ import annotations

import copy
from typing import Dict, Literal

import torch
import torch.nn as nn

from onedp.schedulers.ddpm import DDPMDistillationScheduler
from onedp.schedulers.edm import EDMDistillationScheduler

# import the upstream ConditionalUnet1D via the diffusion_policy
try:
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
except ImportError as e:
    raise ImportError(
        "diffusion_policy is not found on PYTHONPATH.\n"
        "  git clone https://github.com/real-stanford/diffusion_policy.git\n"
        "  export PYTHONPATH=$PYTHONPATH:/path/to/diffusion_policy"
    ) from e



# Generator
class OneDPGenerator(nn.Module):
    """
    Single-step action generator  G_θ(z, O).

    Wraps a ConditionalUnet1D noise-prediction network.  Since the network
    was initialised from a denoising model it still receives a timestep
    embedding; we fix that timestep to `t_init` (65 for DDPM, or the
    σ-equivalent for EDM).

    The clean action is recovered via:
        pred_noise = F_θ(z, t_init, obs_feat)
        A = (z − σ_{t_init} · pred_noise) / α_{t_init}    [DDPM x₀ formula]

    For OneDP-D (deterministic) z is set to zeros, removing stochasticity.
    """

    def __init__(
        self,
        noise_pred_net: ConditionalUnet1D,
        scheduler: DDPMDistillationScheduler | EDMDistillationScheduler,
        t_init: int = 65,             # fixed timestep embedding (DDPM) or idx
        stochastic: bool = True,
    ):
        super().__init__()
        self.noise_pred_net = noise_pred_net
        self.scheduler = scheduler
        self.t_init = t_init
        self.stochastic = stochastic
        self._is_edm = isinstance(scheduler, EDMDistillationScheduler)

 

    def forward(
        self,
        obs_features: torch.Tensor,                  # (B, obs_cond_dim)
        z: torch.Tensor | None = None,               # (B, T_pred, action_dim)
    ) -> torch.Tensor:
        """
        Generate a clean action chunk from noise z (or zeros) and observations.

        Args:
            obs_features:  (B, obs_cond_dim)  already-encoded observation
            z:             (B, T_pred, action_dim)  input noise;
                           sampled automatically if None and stochastic=True;
                           set to zeros for deterministic variant.
        Returns:
            action:  (B, T_pred, action_dim)  clean action chunk
        """
        B = obs_features.shape[0]

        if z is None:
            raise ValueError(
                "z must be provided.  Use `sample_z` to draw it first."
            )

        if self._is_edm:
            # EDM branch
            sigma = self.scheduler.generator_sigma(B, obs_features.device)  # (B,)
            x_sigma = z * sigma.view(-1, 1, 1)

            # EDM preconditioning
            sample_in = self.scheduler.precondition_input(x_sigma, sigma)

            # timestep embeddin
            t_emb = self.scheduler.c_noise(sigma)  # (B,) float

            # Network forward
            F_out = self.noise_pred_net(
                sample=sample_in,
                timestep=t_emb,
                global_cond=obs_features,
            )

            # Decode
            action = self.scheduler.precondition_output(x_sigma, sigma, F_out)
        else:
            # DDPM branch
            # Fixed integer timestep embedding
            t = torch.full(
                (B,), self.t_init, device=obs_features.device, dtype=torch.long
            )

            # Noise prediction
            pred_noise = self.noise_pred_net(
                sample=z,
                timestep=t,
                global_cond=obs_features,
            )  # (B, T_pred, action_dim)

            # Recover clean action
            action = self.scheduler.predict_x0_from_noise(z, t, pred_noise)

        return action

    @torch.no_grad()
    def sample_z(self, batch_size: int, action_shape: tuple, device) -> torch.Tensor:
        """Sample z ~ N(0,I) for the stochastic generator."""
        if self.stochastic:
            return torch.randn(batch_size, *action_shape, device=device)
        else:
            return torch.zeros(batch_size, *action_shape, device=device)



# Full OneDP model  (generator + optional score network + frozen teacher)

class OneDP(nn.Module):
    """
    One-Step Diffusion Policy distillation model.

    Holds:
      - generator         G_θ        (trained)
      - score_network     π_ψ        (trained, OneDP-S only)
      - pretrained_policy π_φ        (frozen teacher)

    Usage:
        # Build from a pre-trained DiffusionPolicy checkpoint:
        onedp = OneDP.from_pretrained(policy, scheduler, variant='stochastic')

        # Training step (returns dict of scalar losses):
        losses = onedp.compute_loss(obs_features, action_shape)

        # Inference (single-step):
        action = onedp.predict_action(obs_features, action_shape)
    """

    def __init__(
        self,
        generator: OneDPGenerator,
        pretrained_noise_pred: nn.Module,     # frozen π_φ noise predictor
        scheduler: DDPMDistillationScheduler | EDMDistillationScheduler,
        variant: Literal["stochastic", "deterministic"] = "stochastic",
        score_network: ConditionalUnet1D | None = None,  # π_ψ (OneDP-S only)
    ):
        super().__init__()
        self.generator = generator
        self.scheduler = scheduler
        self.variant = variant
        self._is_edm = isinstance(scheduler, EDMDistillationScheduler)

        # Frozen pre-trained teacher 
        self.pretrained_noise_pred = pretrained_noise_pred
        for p in self.pretrained_noise_pred.parameters():
            p.requires_grad_(False)
        self.pretrained_noise_pred.eval()

        # Score network  (stochastic variant only)
        if variant == "stochastic":
            assert score_network is not None, (
                "score_network must be provided for the stochastic variant"
            )
        self.score_network = score_network  # may be None for deterministic


    @classmethod
    def from_pretrained(
        cls,
        pretrained_noise_pred: ConditionalUnet1D,
        scheduler: DDPMDistillationScheduler | EDMDistillationScheduler,
        variant: Literal["stochastic", "deterministic"] = "stochastic",
        t_init: int = 65,
    ) -> "OneDP":
        """
        Build OneDP by deep-copying the pre-trained noise predictor into:
          - the action generator  G_θ  (warm-started from π_φ weights)
          - the score network     π_ψ  (warm-started from π_φ weights, OneDP-S only)

        The teacher π_φ is kept as a separate frozen copy.

        Args:
            pretrained_noise_pred: the ConditionalUnet1D from the trained DP
            scheduler:             noise schedule (DDPM or EDM)
            variant:               'stochastic' (OneDP-S) or 'deterministic' (OneDP-D)
            t_init:                fixed timestep for the generator input (DDPM: 65)
        """
        stochastic = variant == "stochastic"

        # Deep-copy weights for the generator
        gen_net = copy.deepcopy(pretrained_noise_pred)
        generator = OneDPGenerator(
            noise_pred_net=gen_net,
            scheduler=scheduler,
            t_init=t_init,
            stochastic=stochastic,
        )

        # Deep-copy weights for the score network (OneDP-S)
        score_net = copy.deepcopy(pretrained_noise_pred) if stochastic else None

        # Keep original weights frozen as teacher
        frozen_teacher = copy.deepcopy(pretrained_noise_pred)

        return cls(
            generator=generator,
            pretrained_noise_pred=frozen_teacher,
            scheduler=scheduler,
            variant=variant,
            score_network=score_net,
        )

    # 
    # Loss computation

    def compute_loss(
        self,
        obs_features: torch.Tensor,   # (B, obs_cond_dim)
        action_shape: tuple,           # (T_pred, action_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation losses for one training step.

        Returns a dict with keys:
          'loss_generator'    — gradient signal for G_θ
          'loss_score_net'    — denoising loss for π_ψ  (OneDP-S only)
          'loss_total'        — sum of the above

        Callers should call `.backward()` on 'loss_total' (or on
        'loss_generator' and 'loss_score_net' separately when using
        separate optimisers).
        """
        if self.variant == "stochastic":
            return self._loss_stochastic(obs_features, action_shape)
        else:
            return self._loss_deterministic(obs_features, action_shape)


    def _loss_stochastic(
        self,
        obs_features: torch.Tensor,
        action_shape: tuple,
    ) -> Dict[str, torch.Tensor]:
        B = obs_features.shape[0]
        device = obs_features.device

        # generate clean action from generator
        z = self.generator.sample_z(B, action_shape, device)    
        A_gen = self.generator(obs_features, z)                  

        # sample diffusion timestep / noise level ─────────────
        if self._is_edm:
            sigma = self.scheduler.sample_sigmas(B, device)      # (B,)
            A_k, noise = self.scheduler.q_sample(A_gen.detach(), sigma)
        else:
            k = self.scheduler.sample_timesteps(B, device)       # (B,) ints
            A_k, noise = self.scheduler.q_sample(A_gen.detach(), k)

        #  update score network 
        if self._is_edm:

            t_emb = self.scheduler.c_noise(sigma)
            # EDM: pass preconditioned input c_in(σ) · A^k
            score_in = self.scheduler.precondition_input(A_k, sigma)
            pred_noise_psi = self.score_network(
                sample=score_in, timestep=t_emb, global_cond=obs_features
            )
            lam = self.scheduler.loss_weight(sigma).view(B, 1, 1)
        else:
            pred_noise_psi = self.score_network(
                sample=A_k, timestep=k, global_cond=obs_features
            )
            lam = self.scheduler.distillation_weight(k) ** 2  # σ_k² = w(k)

        loss_score_net = (lam * (pred_noise_psi - noise) ** 2).mean()

        #  update generator 

        # Recompute A_k with gradient through A_gen
        if self._is_edm:
            A_k_grad, _ = self.scheduler.q_sample(A_gen, sigma, noise)
        else:
            A_k_grad, _ = self.scheduler.q_sample(A_gen, k, noise)

        with torch.no_grad():
            if self._is_edm:
                # score_in and t_emb were computed above (float, no .long())
                eps_phi = self.pretrained_noise_pred(
                    sample=score_in, timestep=t_emb, global_cond=obs_features
                )
                eps_psi = pred_noise_psi  # already computed above
                weight = self.scheduler.distillation_weight(sigma)  
            else:
                eps_phi = self.pretrained_noise_pred(
                    sample=A_k, timestep=k, global_cond=obs_features
                )
                eps_psi = pred_noise_psi
                weight = self.scheduler.distillation_weight(k)       

            # score_diff
            score_diff = weight * (eps_phi - eps_psi)


        loss_gen = (score_diff * A_k_grad).mean()

        return {
            "loss_generator": loss_gen,
            "loss_score_net": loss_score_net,
            "loss_total": loss_gen + loss_score_net,
        }

    # OneDP-D  

    def _loss_deterministic(
        self,
        obs_features: torch.Tensor,
        action_shape: tuple,
    ) -> Dict[str, torch.Tensor]:
        B = obs_features.shape[0]
        device = obs_features.device

        #  generate clean action
        z = self.generator.sample_z(B, action_shape, device)   # zeros
        A_gen = self.generator(obs_features, z)                 # (B, T, D)

        # sample diffusion timestep / noise 
        if self._is_edm:
            sigma = self.scheduler.sample_sigmas(B, device)
            A_k_grad, noise = self.scheduler.q_sample(A_gen, sigma)
        else:
            k = self.scheduler.sample_timesteps(B, device)
            A_k_grad, noise = self.scheduler.q_sample(A_gen, k)

    

        with torch.no_grad():
            if self._is_edm:
                # Float noise embedding
                t_emb = self.scheduler.c_noise(sigma)
                score_in = self.scheduler.precondition_input(A_k_grad.detach(), sigma)
                eps_phi = self.pretrained_noise_pred(
                    sample=score_in, timestep=t_emb, global_cond=obs_features
                )
                weight = self.scheduler.distillation_weight(sigma)   # σ
            else:
                eps_phi = self.pretrained_noise_pred(
                    sample=A_k_grad.detach(), timestep=k, global_cond=obs_features
                )
                weight = self.scheduler.distillation_weight(k)       # σ_k

            score_diff = weight * (eps_phi - noise)   # (B, 1, 1) × (B, T, D)

        # Pseudo-loss with gradient through A_k_grad → A_gen → G_θ
        loss_gen = (score_diff * A_k_grad).mean()

        return {
            "loss_generator": loss_gen,
            "loss_score_net": torch.zeros(1, device=device),
            "loss_total": loss_gen,
        }

   
    # Inference
    @torch.no_grad()
    def predict_action(
        self,
        obs_features: torch.Tensor,   # (B, obs_cond_dim)
        action_shape: tuple,           # (T_pred, action_dim)
    ) -> torch.Tensor:
        """
        Single-step action prediction.

        Args:
            obs_features:  (B, obs_cond_dim)  encoded observations
            action_shape:  (T_pred, action_dim)
        Returns:
            action:  (B, T_pred, action_dim)
        """
        B = obs_features.shape[0]
        device = obs_features.device
        z = self.generator.sample_z(B, action_shape, device)
        return self.generator(obs_features, z)


    # Checkpoint helpers
    def save_checkpoint(self, path, epoch: int, **extra):
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "generator": self.generator.state_dict(),
                "score_network": (
                    self.score_network.state_dict()
                    if self.score_network is not None else None
                ),
                **extra,
            },
            path,
        )

    def load_checkpoint(self, path, strict: bool = True) -> int:
        ckpt = torch.load(path, map_location="cpu")
        self.generator.load_state_dict(ckpt["generator"], strict=strict)
        if self.score_network is not None and ckpt.get("score_network"):
            self.score_network.load_state_dict(ckpt["score_network"], strict=strict)
        return ckpt.get("epoch", 0)
