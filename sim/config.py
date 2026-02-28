"""Configuration: load sim.yaml into typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import yaml


@dataclass
class TimelineConfig:
    duration_s: float
    lidar_rate_hz: float
    camera_rate_hz: float
    tf_rate_hz: float


@dataclass
class ActorConfig:
    speed_mps: float
    waypoints: List[Tuple[float, float]]  # [(Y, X), ...]


@dataclass
class DynamicAgentConfig:
    enabled: bool
    asset_name: str
    speed_mps: float
    waypoints: List[Tuple[float, float]]  # [(Y, X), ...]
    frame_id: str
    start_delay_s: float = 0.0


@dataclass
class MountConfig:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class LidarRenderConfig:
    min_range_m: float
    max_range_m: float


@dataclass
class CameraRenderConfig:
    min_range_m: float
    max_range_m: float
    splat_radius_px: int
    image_format: str = "png"  # "png" or "jpeg"
    car_splat_radius_px: int = 3
    supersample: int = 1  # 1 = normal, 2 = render at 2x then downsample


@dataclass
class FrameNames:
    fixed: str
    ego_body: str
    lidar: str
    camera_body: str
    camera_optical: str


@dataclass
class SimConfig:
    timeline: TimelineConfig
    ego: ActorConfig
    sensor_mounts: dict  # name -> MountConfig
    camera_intrinsics: CameraIntrinsics
    lidar_render: LidarRenderConfig
    camera_render: CameraRenderConfig
    frame_names: FrameNames
    data_dir: Path
    agents: dict[str, DynamicAgentConfig] = field(default_factory=dict)


def _validate(cfg: SimConfig) -> None:
    """Validate config values. Raises ValueError on bad input."""
    li, ca, intr = cfg.lidar_render, cfg.camera_render, cfg.camera_intrinsics
    if not (li.min_range_m > 0 and li.max_range_m > li.min_range_m):
        raise ValueError(f"Bad lidar range: min={li.min_range_m}, max={li.max_range_m}")
    if not (ca.min_range_m > 0 and ca.max_range_m > ca.min_range_m):
        raise ValueError(f"Bad camera range: min={ca.min_range_m}, max={ca.max_range_m}")
    if ca.splat_radius_px < 1 or ca.car_splat_radius_px < 1:
        raise ValueError(f"Splat radius must be >= 1: scene={ca.splat_radius_px}, car={ca.car_splat_radius_px}")
    if ca.image_format not in ("png", "jpeg"):
        raise ValueError(f"Unsupported image format: {ca.image_format!r}")
    if ca.supersample not in (1, 2):
        raise ValueError(f"Supersample must be 1 or 2, got {ca.supersample}")
    if intr.width <= 0 or intr.height <= 0:
        raise ValueError(f"Image dimensions must be positive: {intr.width}x{intr.height}")
    if intr.fx <= 0 or intr.fy <= 0:
        raise ValueError(f"Focal lengths must be positive: fx={intr.fx}, fy={intr.fy}")
    if not (0 <= intr.cx <= intr.width and 0 <= intr.cy <= intr.height):
        raise ValueError(f"Principal point out of bounds: cx={intr.cx}, cy={intr.cy}")


def load_config(path: str | Path) -> SimConfig:
    """Load sim.yaml and return a fully typed SimConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Parse each section into dataclasses so later code gets typed fields.
    timeline = TimelineConfig(**raw["timeline"])

    def parse_actor(d: dict) -> ActorConfig:
        # Preserve sim.yaml waypoint convention: [Y, X].
        return ActorConfig(
            speed_mps=d["speed_mps"],
            waypoints=[(wp[0], wp[1]) for wp in d["waypoints"]],
        )

    def parse_dynamic_agent(name: str, d: dict) -> DynamicAgentConfig:
        return DynamicAgentConfig(
            enabled=bool(d.get("enabled", True)),
            asset_name=str(d["asset_name"]),
            speed_mps=float(d["speed_mps"]),
            waypoints=[(wp[0], wp[1]) for wp in d["waypoints"]],
            frame_id=str(d.get("frame_id", name)),
            start_delay_s=float(d.get("start_delay_s", 0.0)),
        )

    ego = parse_actor(raw["ego"])
    agents = {
        name: parse_dynamic_agent(name, d)
        for name, d in (raw.get("agents") or {}).items()
    }

    mounts = {}
    # Sensor mounts are keyed by sensor name in sim.yaml.
    for name, m in raw["sensor_mounts"].items():
        mounts[name] = MountConfig(**m)

    cam = raw["camera_intrinsics"]
    intrinsics = CameraIntrinsics(
        width=int(cam["width"]), height=int(cam["height"]),
        fx=cam["fx"], fy=cam["fy"], cx=cam["cx"], cy=cam["cy"],
    )
    lidar_render = LidarRenderConfig(**raw["lidar_render"])
    camera_render = CameraRenderConfig(**raw["camera_render"])
    frames = FrameNames(**raw["frame_names"])

    # Build one final config object and validate it before returning.
    cfg = SimConfig(
        timeline=timeline,
        ego=ego,
        agents=agents,
        sensor_mounts=mounts,
        camera_intrinsics=intrinsics,
        lidar_render=lidar_render,
        camera_render=camera_render,
        frame_names=frames,
        data_dir=Path(raw["data_dir"]),
    )
    _validate(cfg)
    return cfg
