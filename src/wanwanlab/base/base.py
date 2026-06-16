from dataclasses import dataclass




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