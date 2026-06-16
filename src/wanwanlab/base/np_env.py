from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class NpEnvState:
    obs: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict[str, Any]
    final_observation: dict[str, np.ndarray] | None = None

    def replace(self, **updates: Any) -> "NpEnvState":
        return dataclasses.replace(self, **updates)


