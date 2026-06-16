import abc
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
from .scene import SceneCfg


@dataclass(frozen=True)
class EnvPlayCapabilities:
    """Env-facing play/render capabilities consumed by training entrypoints"""

    supports_native_interactive_rander: bool = False
    supports_physics_state_playback: bool = False
    supports_native_video_capture: bool = False

@dataclass
class EnvCfg:
    """Config for the environment"""

    scene: SceneCfg | None = None
    sim_dt: float = 0.01
    max_episode_seconds: Optional[float] = None
    ctrl_dt: float = 0.01
    render_spacing: float = 1.0
    render_offset_mode: str = 'grid'
    motrix_max_iterations: Optional[int] = None
    post_step_forward_sensor: bool = False


    @property
    def max_episode_steps(self) -> Optional[int]:
        """
        return the max episode steps
        """
        if self.max_episode_seconds is None:
            return None
        return int(self.max_episode_seconds / self.ctrl_dt)

    @property
    def sim_substeps(self) -> int:
        """
        return the number of simulation steps per control step
        """
        return int(round(self.ctrl_dt / self.sim_dt))

    def validate(self):
        """
        validate the config
        """
        if self.sim_dt > self.ctrl_dt:
            raise ValueError("sim_dt must be less than or equal to ctrl_dt")

class ABEnv(abc.ABC):
    @property
    def play_capabilities(self) -> EnvPlayCapabilities:
        """Return env-facing play/render capabilities."""
        return EnvPlayCapabilities()
    
