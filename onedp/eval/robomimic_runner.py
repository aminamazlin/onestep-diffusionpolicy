"""
Robomimic simulation evaluator for OneDP / DiffusionPolicy.

Runs rollout episodes in a Robomimic simulation environment and reports
success rate and completion time — the primary metrics from the paper
(Section 3.1, Table 1 & 2).

Supports all tasks used in the paper:
  Square-mh / Square-ph, ToolHang-ph, Transport-mh / Transport-ph

Evaluation protocol (paper Section 3.1):
  - 100 distinct initial conditions per evaluation
  - Action chunking: predict `horizon` actions, execute `n_action_steps`
  - Success rate as primary metric; mean completion steps for successes

The ``predict_fn`` interface expected by :meth:`RobomimicEvaluator.evaluate`::

    def predict_fn(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        # obs_dict: {
        #   image_key:  (n_obs, C, H, W) float32 in [0, 1]
        #   lowdim_key: (n_obs, dim)     float32
        # }
        # returns:  (n_action_steps, action_dim)  float32  (raw env scale)
"""

from __future__ import annotations

import numpy as np
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional

import h5py


# ── Per-task step budgets (generous upper bound; follow paper Appendix A) ──────
_TASK_MAX_STEPS: Dict[str, int] = {
    "lift":       400,
    "can":        400,
    "square":     400,
    "transport":  700,
    "tool_hang":  700,
    "toolhang":   700,
}
_DEFAULT_MAX_STEPS = 400


def max_steps_for_dataset(dataset_path: str) -> int:
    """Return a sensible step budget inferred from the dataset filename."""
    p = dataset_path.lower()
    for keyword, steps in _TASK_MAX_STEPS.items():
        if keyword in p:
            return steps
    return _DEFAULT_MAX_STEPS


class RobomimicEvaluator:
    """
    Evaluate a visuomotor policy on a Robomimic simulation task.

    Args:
        dataset_path:     Path to the Robomimic HDF5 dataset.
                          Used to (a) extract environment metadata and
                          (b) load deterministic initial simulator states.
        image_keys:       Image observation keys to pass to the policy.
        lowdim_keys:      Low-dim observation keys to pass to the policy.
        n_obs_steps:      Number of frames in the observation window.
        n_action_steps:   Number of actions executed per prediction call.
        max_steps:        Hard step limit per episode (auto-detected if 0).
        n_eval_episodes:  Number of rollout episodes (paper: 100).

    Example::

        evaluator = RobomimicEvaluator("data/square_ph.hdf5")
        metrics = evaluator.evaluate(predict_fn)
        # {'success_rate': 0.92, 'mean_completion_steps': 120, ...}
    """

    def __init__(
        self,
        dataset_path: str | Path,
        image_keys: tuple = ("agentview_image", "robot0_eye_in_hand_image"),
        lowdim_keys: tuple = (),
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        max_steps: int = 0,
        n_eval_episodes: int = 100,
    ):
        self.dataset_path = str(dataset_path)
        self.image_keys = list(image_keys)
        self.lowdim_keys = list(lowdim_keys)
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps or max_steps_for_dataset(self.dataset_path)
        self.n_eval_episodes = n_eval_episodes

        self._initial_states: List[np.ndarray] = []
        self._load_initial_states()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_initial_states(self) -> None:
        """
        Load the first simulator state of every demonstration.

        These are used to reset the environment deterministically so that
        each evaluation episode starts from a known, reproducible state —
        matching the paper's "100 different initial conditions" protocol.
        """
        with h5py.File(self.dataset_path, "r") as f:
            for dk in sorted(f["data"].keys()):
                demo = f["data"][dk]
                if "states" in demo and len(demo["states"]) > 0:
                    self._initial_states.append(demo["states"][0])

        if not self._initial_states:
            raise RuntimeError(
                f"No simulator states found in {self.dataset_path}.\n"
                "Robomimic datasets must include the 'states' array for "
                "deterministic evaluation.  Re-collect data with "
                "hdf5_filter.py or robomimic's dataset_states_to_obs.py."
            )

    def _create_env(self):
        """Create a Robomimic simulation environment from dataset metadata."""
        try:
            import robomimic.utils.file_utils as FileUtils
            import robomimic.utils.env_utils as EnvUtils
        except ImportError as e:
            raise ImportError(
                "robomimic is required for simulation evaluation.\n"
                "  pip install robomimic\n"
                f"Original error: {e}"
            ) from e

        env_meta = FileUtils.get_env_metadata_from_dataset(self.dataset_path)
        env = EnvUtils.create_env_from_metadata(
            env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=True,
        )
        return env

    # ------------------------------------------------------------------
    # Observation formatting
    # ------------------------------------------------------------------

    def _stack_obs(self, obs_deque: deque) -> Dict[str, np.ndarray]:
        """
        Convert an observation deque to the format expected by the policy.

        Robomimic env returns raw observations (images as (H, W, C) uint8).
        This method converts them to the policy-ready format used during
        training:

        - Images:  (n_obs, H, W, C) uint8  →  (n_obs, C, H, W) float32/255
        - Low-dim: (n_obs, dim)     float  →  (n_obs, dim)     float32
        """
        obs_list = list(obs_deque)  # oldest → newest
        result: Dict[str, np.ndarray] = {}

        for key in self.image_keys:
            frames = [o[key] for o in obs_list if key in o]
            if not frames:
                continue
            arr = np.stack(frames, axis=0)             # (n_obs, H, W, C)
            arr = arr.transpose(0, 3, 1, 2)            # (n_obs, C, H, W)
            result[key] = arr.astype(np.float32) / 255.0

        for key in self.lowdim_keys:
            frames = [o[key] for o in obs_list if key in o]
            if not frames:
                continue
            result[key] = np.stack(frames, axis=0).astype(np.float32)  # (n_obs, dim)

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
            predict_fn: A callable with signature::

                obs_dict → action_array

                where obs_dict has keys matching image_keys / lowdim_keys
                (images: (n_obs, C, H, W) float32 [0,1];
                 lowdim: (n_obs, dim) float32)
                and action_array has shape (n_action_steps, action_dim).

            verbose: Print per-episode results.

        Returns:
            Dict with keys:
              - ``success_rate``         fraction of successful episodes
              - ``mean_completion_steps`` avg steps for successful episodes
              - ``n_success``            raw count of successes
              - ``n_episodes``           total episodes evaluated
        """
        env = self._create_env()
        n_success = 0
        completion_steps: List[int] = []

        for ep_idx in range(self.n_eval_episodes):
            # Cycle through available initial states if fewer than n_eval_episodes
            state = self._initial_states[ep_idx % len(self._initial_states)]

            # Deterministic environment reset
            env.reset()
            env.reset_to({"states": state})

            # Grab initial observation and pre-fill the sliding window
            obs = env.get_observation()
            obs_deque: deque = deque(
                [obs] * self.n_obs_steps, maxlen=self.n_obs_steps
            )

            success = False
            step_count = 0

            while step_count < self.max_steps:
                # ── Predict action chunk ──────────────────────────────────
                obs_dict = self._stack_obs(obs_deque)
                actions = predict_fn(obs_dict)  # (n_action_steps, action_dim)

                # ── Execute each action in the chunk ──────────────────────
                for action in actions:
                    obs, _reward, done, info = env.step(action)
                    obs_deque.append(obs)
                    step_count += 1

                    # Robomimic success check (works across all suite tasks)
                    if hasattr(env, "is_success"):
                        success = bool(env.is_success().get("task", False))
                    else:
                        success = bool(info.get("success", False))

                    if success or done or step_count >= self.max_steps:
                        break

                if success or done or step_count >= self.max_steps:
                    break

            if success:
                n_success += 1
                completion_steps.append(step_count)

            if verbose:
                status = "SUCCESS" if success else "FAIL"
                print(f"  Episode {ep_idx+1:3d}/{self.n_eval_episodes}: "
                      f"{status}  steps={step_count}")

        env.close()

        success_rate = n_success / self.n_eval_episodes
        mean_steps = (
            float(np.mean(completion_steps))
            if completion_steps
            else float(self.max_steps)
        )

        return {
            "success_rate": success_rate,
            "mean_completion_steps": mean_steps,
            "n_success": n_success,
            "n_episodes": self.n_eval_episodes,
        }
