import math
from typing import Any, cast

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

_LOG_2_PI = math.log(2.0 * math.pi)
_NORMAL_ENTROPY_OFFSET = 0.5 * (1.0 + _LOG_2_PI)


class FinalObservationAwarePPO(PPO):
    """PPO variant that bootstraps time limits from env final_observation."""

    learning_rate: float

    def __init__(
        self,
        *args: Any,
        enable_compile: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.enable_compile = (
            bool(enable_compile)
            and torch.device(self.device).type == "cuda"
            and hasattr(torch, "compile")
        )
        self._minibatch_loss_fn = self._minibatch_loss_tensors
        if self.enable_compile:
            self._compile_training_methods()