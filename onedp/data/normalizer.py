"""
Linear normaliser that maps data to [-1, 1] (actions) or [0, 1] (images).

Matches the normalisation convention used in the upstream diffusion_policy
library so that pre-trained policy checkpoints can be used directly.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, Union

TensorOrArray = Union[torch.Tensor, np.ndarray]


class LinearNormalizer:
    """
    Per-key linear normaliser.  Stores (min, max) statistics fitted on the
    training dataset and maps values to the target range.

    Typical usage:
        normalizer = LinearNormalizer()
        normalizer.fit({"action": actions, "obs/image": images})

        # In training loop:
        normed = normalizer.normalize({"action": raw_action})
        raw    = normalizer.unnormalize({"action": normed_action})
    """

    def __init__(self):
        self.stats: Dict[str, Dict[str, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        data: Dict[str, TensorOrArray],
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
    ) -> "LinearNormalizer":
        """
        Compute per-key (min, max) statistics from data tensors/arrays.

        Args:
            data:        {key: (N, ...) tensor or array}
            output_max:  target range upper bound  (default  1.0 for actions)
            output_min:  target range lower bound  (default -1.0 for actions)
            range_eps:   minimum range to avoid division by zero
        """
        for key, values in data.items():
            if isinstance(values, np.ndarray):
                values = torch.from_numpy(values).float()
            elif not isinstance(values, torch.Tensor):
                raise TypeError(f"Expected tensor or ndarray for key '{key}'")

            # Flatten all dims except batch
            flat = values.reshape(len(values), -1)
            data_min = flat.min(dim=0).values
            data_max = flat.max(dim=0).values

            # Ensure range is not degenerate
            data_range = (data_max - data_min).clamp(min=range_eps)

            self.stats[key] = {
                "min": data_min,
                "max": data_max,
                "range": data_range,
                "output_min": torch.tensor(output_min),
                "output_max": torch.tensor(output_max),
                "shape": values.shape[1:],  # spatial shape (without batch)
            }
        return self

    # ------------------------------------------------------------------
    # Normalise / unnormalise
    # ------------------------------------------------------------------

    def normalize(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {k: self._normalize_single(k, v) for k, v in data.items()}

    def unnormalize(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {k: self._unnormalize_single(k, v) for k, v in data.items()}

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self._normalize_single("action", action)

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self._unnormalize_single("action", action)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize_single(self, key: str, x: torch.Tensor) -> torch.Tensor:
        stat = self._get_stat(key, x)
        # x_norm = (x − min) / range * (out_max − out_min) + out_min
        x_flat = x.reshape(*x.shape[: x.dim() - len(stat["shape"])], -1)
        x_norm = (x_flat - stat["min"].to(x)) / stat["range"].to(x)
        out_range = stat["output_max"] - stat["output_min"]
        x_norm = x_norm * out_range.to(x) + stat["output_min"].to(x)
        return x_norm.reshape_as(x)

    def _unnormalize_single(self, key: str, x: torch.Tensor) -> torch.Tensor:
        stat = self._get_stat(key, x)
        out_range = stat["output_max"] - stat["output_min"]
        x_flat = x.reshape(*x.shape[: x.dim() - len(stat["shape"])], -1)
        x_raw = (x_flat - stat["output_min"].to(x)) / out_range.to(x)
        x_raw = x_raw * stat["range"].to(x) + stat["min"].to(x)
        return x_raw.reshape_as(x)

    def _get_stat(self, key: str, x: torch.Tensor) -> dict:
        if key not in self.stats:
            raise KeyError(
                f"No normalisation statistics for key '{key}'. "
                f"Available keys: {list(self.stats.keys())}"
            )
        return self.stats[key]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {k: {sk: sv for sk, sv in v.items()} for k, v in self.stats.items()}

    def load_state_dict(self, state: dict) -> "LinearNormalizer":
        self.stats = state
        return self

    def to(self, device) -> "LinearNormalizer":
        for key in self.stats:
            for sk in self.stats[key]:
                if isinstance(self.stats[key][sk], torch.Tensor):
                    self.stats[key][sk] = self.stats[key][sk].to(device)
        return self
