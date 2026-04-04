"""
PushT simulation evaluator for OneDP / DiffusionPolicy.

The PushT task (adapted from IBC, introduced by Chi et al. 2023) involves
pushing a T-shaped block into a fixed target region using a circular
end-effector.  It is included in the paper's simulation benchmark (Table 1).

Metric: ``mean_coverage`` — the maximum fraction of the T-block's area
that overlapped the target region during the episode, averaged over all
evaluation episodes.  (Not a binary success flag.)

Unlike the Robomimic tasks, PushT does not need pre-recorded initial states —
episodes are seeded from a deterministic integer seed range, matching the
evaluation protocol used in the upstream diffusion_policy repository.

The ``predict_fn`` interface::

    def predict_fn(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        # obs_dict: {
        #   'image':     (n_obs, C, H, W)  float32 in [0, 1]
        #   'agent_pos': (n_obs, 2)         float32  (optional)
        # }
        # returns: (n_action_steps, 2)  float32  (agent xy in [0, 512])
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Callable, Dict, List


class PushTEvaluator:
    """
    Evaluate a visuomotor policy on the PushT simulation task.

    Args:
        n_obs_steps:     Number of frames in the observation window.
        n_action_steps:  Number of actions executed per prediction call.
        max_steps:       Hard step limit per episode.
        n_eval_episodes: Number of rollout episodes.
        render_size:     Image size used by PushTImageEnv (paper: 96).
        start_seed:      First seed for episode resets (ensures reproducibility).

    Example::

        evaluator = PushTEvaluator(n_eval_episodes=100)
        metrics = evaluator.evaluate(predict_fn)
        # {'mean_coverage': 0.863, 'max_coverage': 1.0, ...}
    """

    def __init__(
        self,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        max_steps: int = 300,
        n_eval_episodes: int = 100,
        render_size: int = 96,
        start_seed: int = 100_000,
    ):
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.n_eval_episodes = n_eval_episodes
        self.render_size = render_size
        self.start_seed = start_seed

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def _create_env(self):
        """
        Create a PushTImageEnv from the diffusion_policy package.

        PushTImageEnv returns:
          obs: {'image': (H, W, C) uint8, 'agent_pos': (2,) float}
          info: {'coverage': float}  at each step
        """
        try:
            from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
        except ImportError as e:
            raise ImportError(
                "diffusion_policy.env.pusht is required for PushT evaluation.\n"
                "  git clone https://github.com/real-stanford/diffusion_policy.git\n"
                "  export PYTHONPATH=$PYTHONPATH:/path/to/diffusion_policy\n"
                f"Original error: {e}"
            ) from e

        return PushTImageEnv(render_size=self.render_size)

    # ------------------------------------------------------------------
    # Observation formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_raw_obs(raw_obs) -> Dict[str, np.ndarray]:
        """
        Normalise a single raw observation from PushTImageEnv.

        PushTImageEnv may return a plain ndarray (image only) or a dict.
        Normalise to a consistent dict with float32 arrays.
        """
        if isinstance(raw_obs, np.ndarray):
            return {"image": raw_obs}
        obs: Dict[str, np.ndarray] = {}
        if "image" in raw_obs:
            obs["image"] = np.asarray(raw_obs["image"])
        if "agent_pos" in raw_obs:
            obs["agent_pos"] = np.asarray(raw_obs["agent_pos"], dtype=np.float32)
        return obs

    def _stack_obs(self, obs_deque: deque) -> Dict[str, np.ndarray]:
        """
        Convert the observation deque to policy-ready format.

        - image:     (n_obs, H, W, C) uint8  →  (n_obs, C, H, W) float32/255
        - agent_pos: (n_obs, 2)              →  (n_obs, 2)        float32
        """
        obs_list = list(obs_deque)
        result: Dict[str, np.ndarray] = {}

        if "image" in obs_list[0]:
            frames = [o["image"] for o in obs_list]
            arr = np.stack(frames, axis=0)          # (n_obs, H, W, C)
            arr = arr.transpose(0, 3, 1, 2)         # (n_obs, C, H, W)
            result["image"] = arr.astype(np.float32) / 255.0

        if "agent_pos" in obs_list[0]:
            result["agent_pos"] = np.stack(
                [o["agent_pos"] for o in obs_list], axis=0
            ).astype(np.float32)  # (n_obs, 2)

        return result

    # ------------------------------------------------------------------
    # Rollout evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        predict_fn: Callable[[Dict[str, np.ndarray]], np.ndarray],
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Run policy rollout evaluation.

        Args:
            predict_fn: callable(obs_dict) → (n_action_steps, 2) float32 action.
            verbose:    Print per-episode coverage values.

        Returns:
            Dict with keys:
              - ``mean_coverage``  average max-coverage across episodes (primary)
              - ``max_coverage``   best episode coverage
              - ``min_coverage``   worst episode coverage
              - ``n_episodes``     total episodes evaluated
        """
        env = self._create_env()
        episode_coverages: List[float] = []

        for ep_idx in range(self.n_eval_episodes):
            seed = self.start_seed + ep_idx

            # Reset with deterministic seed
            reset_result = env.reset(seed=seed)
            # Handle both old gym (obs,) and new gym (obs, info)
            if isinstance(reset_result, tuple):
                raw_obs = reset_result[0]
            else:
                raw_obs = reset_result

            obs = self._normalize_raw_obs(raw_obs)
            obs_deque: deque = deque(
                [obs] * self.n_obs_steps, maxlen=self.n_obs_steps
            )

            max_coverage = 0.0
            step_count = 0
            done = False

            while not done and step_count < self.max_steps:
                # ── Predict action chunk ──────────────────────────────────
                obs_dict = self._stack_obs(obs_deque)
                actions = predict_fn(obs_dict)  # (n_action_steps, 2)

                # ── Execute each action in the chunk ──────────────────────
                for action in actions:
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        # Gymnasium API: (obs, reward, terminated, truncated, info)
                        raw_obs, _reward, terminated, truncated, info = step_result
                        done = bool(terminated) or bool(truncated)
                    else:
                        # Old gym API: (obs, reward, done, info)
                        raw_obs, _reward, done, info = step_result

                    obs = self._normalize_raw_obs(raw_obs)
                    obs_deque.append(obs)
                    step_count += 1

                    # PushT coverage: fraction of T-block area in target
                    coverage = float(info.get("coverage", 0.0))
                    if coverage > max_coverage:
                        max_coverage = coverage

                    if done or step_count >= self.max_steps:
                        break

            episode_coverages.append(max_coverage)

            if verbose:
                print(f"  Episode {ep_idx+1:3d}/{self.n_eval_episodes}: "
                      f"coverage={max_coverage:.3f}")

        env.close()

        return {
            "mean_coverage": float(np.mean(episode_coverages)),
            "max_coverage":  float(np.max(episode_coverages)),
            "min_coverage":  float(np.min(episode_coverages)),
            "n_episodes":    self.n_eval_episodes,
        }
