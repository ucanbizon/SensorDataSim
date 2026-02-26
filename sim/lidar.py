"""VLP-16 LiDAR renderer: spherical depth-buffer approach.

Simulates a Velodyne VLP-16 by:
1. Transforming scene points into the velodyne frame
2. Converting to spherical coordinates (azimuth, elevation, range)
3. Assigning each point to the nearest VLP-16 ring (with tolerance gate)
4. Keeping the closest point per (ring, azimuth_bin) cell
5. Returning the original winning points (no bin-center reconstruction)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LidarRenderConfig
from .numba_kernels import lidar_scatter_min
from .scene_compose import ComposedScene
from .transforms import invert_se3, transform_points

# -- VLP-16 hardware specs --
VLP16_ELEVATION_DEG = np.array([
    -15, -13, -11, -9, -7, -5, -3, -1,
      1,   3,   5,  7,  9, 11, 13, 15,
], dtype=np.float64)
VLP16_NUM_RINGS = 16
VLP16_RING_SPACING_DEG = 2.0
VLP16_ELEVATION_TOLERANCE_DEG = VLP16_RING_SPACING_DEG / 2.0
VLP16_AZIMUTH_RESOLUTION_DEG = 0.2
VLP16_NUM_AZIMUTH_BINS = int(360.0 / VLP16_AZIMUTH_RESOLUTION_DEG)  # 1800
VLP16_AZIMUTH_BIN_WIDTH_RAD = np.deg2rad(VLP16_AZIMUTH_RESOLUTION_DEG)
_N_CELLS = VLP16_NUM_RINGS * VLP16_NUM_AZIMUTH_BINS  # 28800

# Intensity remap for Foxglove visualization:
# - static/background points are compressed into a darker band
# - dynamic agents get a bright constant intensity so they pop when coloring by intensity
_STATIC_INTENSITY_VIS_MAX = 40.0
_AGENT_INTENSITY_VIS = 255.0

# -- Reusable scatter-min buffers --
_best_range: np.ndarray | None = None
_best_idx: np.ndarray | None = None


@dataclass
class LidarFrame:
    points: np.ndarray       # (K, 3) float32, velodyne frame
    intensity: np.ndarray    # (K,) float32
    ring: np.ndarray         # (K,) uint16, channel index 0-15


@dataclass
class LidarStats:
    n_in_range: int
    n_after_tol: int
    n_hits: int
    hits_per_ring: np.ndarray  # (16,) int
    empty_bin_fraction: float
    range_min: float
    range_max: float


def _get_buffers():
    # Reuse per-cell winner buffers across frames. The LiDAR kernel writes one
    # scalar winner per (ring, azimuth_bin), so the size is fixed.
    global _best_range, _best_idx
    if _best_range is None:
        _best_range = np.empty(_N_CELLS, dtype=np.float64)
        _best_idx = np.empty(_N_CELLS, dtype=np.int64)
    return _best_range, _best_idx


def render_lidar_frame(
    scene: ComposedScene,
    lidar_cfg: LidarRenderConfig,
) -> tuple[LidarFrame, LidarStats]:
    """Render one VLP-16 scan from a composed scene."""
    min_range = lidar_cfg.min_range_m
    max_range = lidar_cfg.max_range_m

    # Gather a single point/intensity array so the kernel sees one contiguous
    # point set. Dynamic agents are already in map frame at this timestamp.
    #
    # Intensity is also used as a visualization channel in Foxglove. To make
    # dynamic agents stand out when "Color By = Intensity", I compress static
    # intensities into a darker band and assign a bright fixed value to agents.
    if scene.agent_points_map.shape[0] == 0:
        all_pts = scene.static_points
        # Keep the same relative static contrast, just map it to a lower range.
        all_int = (scene.static_intensity.astype(np.float32, copy=False) / 255.0) * _STATIC_INTENSITY_VIS_MAX
    else:
        all_pts = np.concatenate([scene.static_points, scene.agent_points_map])
        static_int = (scene.static_intensity.astype(np.float32, copy=False) / 255.0) * _STATIC_INTENSITY_VIS_MAX
        agent_int = np.full(scene.agent_intensity.shape, _AGENT_INTENSITY_VIS, dtype=np.float32)
        all_int = np.concatenate([static_int, agent_int])

    # LiDAR "scatter-min" kernel:
    # - transform each point to velodyne frame
    # - assign nearest ring and azimuth bin
    # - keep the closest range per cell
    #
    # It stores the original source-point index in best_idx so we can gather
    # exact original geometry/colors after winner selection.
    T_vel_map = invert_se3(scene.tf.T_map_velodyne)
    best_range, best_idx = _get_buffers()
    best_range.fill(np.inf)
    best_idx.fill(-1)

    n_in_range, n_after_tol = lidar_scatter_min(
        all_pts,
        np.ascontiguousarray(T_vel_map[:3, :3]),
        np.ascontiguousarray(T_vel_map[:3, 3]),
        min_range ** 2, max_range ** 2,
        float(VLP16_ELEVATION_DEG[0]), VLP16_RING_SPACING_DEG,
        VLP16_ELEVATION_TOLERANCE_DEG,
        VLP16_NUM_RINGS, VLP16_NUM_AZIMUTH_BINS,
        float(VLP16_AZIMUTH_BIN_WIDTH_RAD),
        best_range, best_idx,
    )

    # Gather only cells that received at least one winner point.
    hit_cells = np.nonzero(best_idx >= 0)[0]
    n_hits = hit_cells.size
    if n_hits == 0:
        return (
            LidarFrame(np.zeros((0, 3), np.float32), np.zeros(0, np.float32), np.zeros(0, np.uint16)),
            LidarStats(0, 0, 0, np.zeros(VLP16_NUM_RINGS, np.int64), 1.0, 0.0, 0.0),
        )

    hit_orig = best_idx[hit_cells]
    # Re-transform only the winners into velodyne frame. This keeps the kernel
    # simple and avoids storing transformed coordinates for all input points.
    out_points = transform_points(T_vel_map, all_pts[hit_orig]).astype(np.float32)
    out_intensity = all_int[hit_orig].astype(np.float32)
    # Cell layout is [ring][azimuth_bin], so integer division recovers ring id.
    out_ring = (hit_cells // VLP16_NUM_AZIMUTH_BINS).astype(np.uint16)
    # bincount is faster and simpler than a Python loop over 16 rings.
    hits_per_ring = np.bincount(out_ring, minlength=VLP16_NUM_RINGS)[:VLP16_NUM_RINGS].astype(np.int64)

    return (
        LidarFrame(points=out_points, intensity=out_intensity, ring=out_ring),
        LidarStats(
            n_in_range=n_in_range, n_after_tol=n_after_tol, n_hits=n_hits,
            hits_per_ring=hits_per_ring,
            empty_bin_fraction=1.0 - n_hits / _N_CELLS,
            range_min=float(best_range[hit_cells].min()),
            range_max=float(best_range[hit_cells].max()),
        ),
    )
