"""TF tree: static sensor mounts + per-timestamp snapshot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .transforms import make_T_camera_link_optical, pose_to_matrix


@dataclass
class TFSnapshot:
    """All transforms at a single timestamp. Single source of truth."""

    t: float

    # Dynamic
    T_map_base: np.ndarray        # ego body in map
    T_map_agents: dict[str, np.ndarray]  # dynamic agents by child frame id

    # Static (stored for completeness / MCAP export)
    T_base_velodyne: np.ndarray
    T_base_camera_link: np.ndarray
    T_camera_link_optical: np.ndarray

    # ── Convenience properties (derived, not stored separately) ──

    @property
    def T_map_velodyne(self) -> np.ndarray:
        """VLP-16 origin + orientation in map frame."""
        return self.T_map_base @ self.T_base_velodyne

    @property
    def T_map_camera_optical(self) -> np.ndarray:
        """Camera optical frame in map (Z=forward, X=right, Y=down)."""
        return self.T_map_base @ self.T_base_camera_link @ self.T_camera_link_optical



class TFTree:
    """Manages static sensor mounts and builds TF snapshots."""

    def __init__(self, config: SimConfig):
        # Precompute static mount transforms once. Per-frame snapshots only need
        # to swap in dynamic map->base (and optional agent) transforms.
        vel = config.sensor_mounts["velodyne"]
        self.T_base_velodyne = pose_to_matrix(
            vel.x, vel.y, vel.z, vel.roll, vel.pitch, vel.yaw
        )

        cam = config.sensor_mounts["camera_link"]
        self.T_base_camera_link = pose_to_matrix(
            cam.x, cam.y, cam.z, cam.roll, cam.pitch, cam.yaw
        )

        self.T_camera_link_optical = make_T_camera_link_optical()

    def snapshot(
        self,
        t: float,
        T_map_base: np.ndarray,
        T_map_agents: dict[str, np.ndarray] | None = None,
    ) -> TFSnapshot:
        """Build a complete TF snapshot at time t."""
        return TFSnapshot(
            t=t,
            T_map_base=T_map_base,
            T_map_agents=T_map_agents or {},
            T_base_velodyne=self.T_base_velodyne,
            T_base_camera_link=self.T_base_camera_link,
            T_camera_link_optical=self.T_camera_link_optical,
        )
