"""Environment registry and base classes."""

from wanwanlab.base.final_observation import (
    TerminalObservationContract,
    TransitionBootstrapContract,
    patch_transition_next_obs,
    resolve_terminal_observation_contract,
    resolve_transition_bootstrap_contract,
)
from wanwanlab.base.observations import (
    flatten_obs_dict,
    flatten_policy_obs_dict,
    get_critic_base_dim,
    get_obs_dims,
    split_obs_dict,
)
from wanwanlab.base.registry import ensure_registries

__all__ = [
    "TerminalObservationContract",
    "TransitionBootstrapContract",
    "ensure_registries",
    "flatten_obs_dict",
    "flatten_policy_obs_dict",
    "get_critic_base_dim",
    "get_obs_dims",
    "patch_transition_next_obs",
    "resolve_terminal_observation_contract",
    "resolve_transition_bootstrap_contract",
    "split_obs_dict",
]
