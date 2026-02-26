"""Scene composition: combine static + dynamic geometry at time t."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assets import SceneAssets
from .tf_tree import TFSnapshot
from .transforms import transform_points


_CULL_CACHE_POS_THRESHOLD_M = 2.5
_CULL_CACHE_MAX_AGE_S = 2.0
_CULL_CACHE_QUERY_MARGIN_M = 3.0


@dataclass
class ComposedScene:
    """The full world state at time t. Renderers consume this."""

    t: float
    tf: TFSnapshot

    # Static geometry (references - not copied per frame)
    static_points: np.ndarray      # (N, 3) map frame
    static_colors: np.ndarray      # (N, 3) uint8
    static_labels: np.ndarray      # (N,) int32
    static_intensity: np.ndarray   # (N,) float32

    # Dynamic geometry (transformed per frame)
    agent_points_map: np.ndarray   # (M, 3) car in map frame at time t
    agent_colors: np.ndarray       # (M, 3) uint8
    agent_intensity: np.ndarray    # (M,) float32
    static_car_label_id: int | None = None
    static_cars_fullres_points: np.ndarray | None = None  # optional static-car subset in map frame
    static_cars_fullres_colors: np.ndarray | None = None


@dataclass
class StaticSubsets:
    """Static geometry slices selected around the current ego pose."""

    points: np.ndarray
    colors: np.ndarray
    labels: np.ndarray
    intensity: np.ndarray
    fullres_car_points: np.ndarray | None
    fullres_car_colors: np.ndarray | None


class SpatialCullCache:
    """Last-query cache for XY KD-tree culling.

    Queries a superset (radius + margin) from the KD-tree, caches it,
    and cheaply filters to exact radius each frame.  Refreshes when
    ego moves > threshold or time exceeds max_age.
    """

    def __init__(self):
        self._last_t: float = -1e30
        self._last_xy: np.ndarray | None = None
        self._cached_static_idx: np.ndarray | None = None
        self._cached_static_xy: np.ndarray | None = None
        self._cached_fullres_idx: np.ndarray | None = None
        self._cached_fullres_xy: np.ndarray | None = None

    def _needs_refresh(self, t: float, ego_xy: np.ndarray) -> bool:
        if self._last_xy is None:
            return True
        if t - self._last_t > _CULL_CACHE_MAX_AGE_S:
            return True
        dx = ego_xy[0] - self._last_xy[0]
        dy = ego_xy[1] - self._last_xy[1]
        return (dx * dx + dy * dy) > _CULL_CACHE_POS_THRESHOLD_M ** 2

    def get_indices(self, t: float, ego_xy: np.ndarray,
                    cull_radius_m: float, assets: SceneAssets,
                    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return filtered static/fullres indices for the current ego pose."""
        if self._needs_refresh(t, ego_xy):
            # Query a superset once, then cheap XY squared-distance filtering can
            # be reused for nearby frames while ego motion stays small.
            q_radius = cull_radius_m + _CULL_CACHE_QUERY_MARGIN_M
            self._last_t = t
            self._last_xy = ego_xy.copy()

            idx = assets.static_kd_tree.query_ball_point(ego_xy, r=q_radius)
            self._cached_static_idx = np.fromiter(idx, dtype=np.intp, count=len(idx))
            self._cached_static_xy = (
                assets.static_points[self._cached_static_idx, :2]
                if self._cached_static_idx.size
                else None
            )

            if assets.static_cars_fullres_kd_tree is not None:
                idx = assets.static_cars_fullres_kd_tree.query_ball_point(ego_xy, r=q_radius)
                self._cached_fullres_idx = np.fromiter(idx, dtype=np.intp, count=len(idx))
                self._cached_fullres_xy = (
                    assets.static_cars_fullres_points[self._cached_fullres_idx, :2]
                    if self._cached_fullres_idx.size
                    else None
                )
            else:
                self._cached_fullres_idx = self._cached_fullres_xy = None

        return (
            self._filter(self._cached_static_idx, self._cached_static_xy, ego_xy, cull_radius_m),
            self._filter(self._cached_fullres_idx, self._cached_fullres_xy, ego_xy, cull_radius_m),
        )

    @staticmethod
    def _filter(idx, xy, ego_xy, radius_m):
        if idx is None or idx.size == 0:
            return idx
        dx = xy[:, 0] - ego_xy[0]
        dy = xy[:, 1] - ego_xy[1]
        # Squared distance avoids sqrt in a very common per-frame operation.
        keep = (dx * dx + dy * dy) <= radius_m * radius_m
        return idx[keep]


def _compose_static_subsets(
    t: float,
    assets: SceneAssets,
    ego_xy: np.ndarray,
    cull_radius_m: float,
    cull_cache: SpatialCullCache,
) -> StaticSubsets:
    """Gather static geometry subsets around ego.

    A named return object keeps the call site readable and avoids a fragile
    6-element positional tuple.
    """
    idx, idx_fr = cull_cache.get_indices(t, ego_xy, cull_radius_m, assets)

    static_points = assets.static_points[idx]
    static_colors = assets.static_colors[idx]
    static_labels = assets.static_labels[idx]
    static_intensity = assets.static_intensity[idx]

    if (assets.static_cars_fullres_points is not None
            and assets.static_cars_fullres_colors is not None
            and idx_fr is not None):
        static_cars_fullres_points = assets.static_cars_fullres_points[idx_fr]
        static_cars_fullres_colors = assets.static_cars_fullres_colors[idx_fr]
    else:
        static_cars_fullres_points = None
        static_cars_fullres_colors = None

    return StaticSubsets(
        points=static_points,
        colors=static_colors,
        labels=static_labels,
        intensity=static_intensity,
        fullres_car_points=static_cars_fullres_points,
        fullres_car_colors=static_cars_fullres_colors,
    )


def _empty_agent_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return empty arrays with the dtypes expected by the renderers."""
    return (
        np.zeros((0, 3), dtype=np.float64),
        np.zeros((0, 3), dtype=np.uint8),
        np.zeros(0, dtype=np.float32),
    )


def _compose_dynamic_agents(
    assets: SceneAssets,
    tf: TFSnapshot,
    ego_xy: np.ndarray,
    include_agent: bool,
    cull_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform active dynamic agents into map frame and cull by XY radius."""
    if include_agent and assets.dynamic_agent_assets:
        pts_list: list[np.ndarray] = []
        col_list: list[np.ndarray] = []
        int_list: list[np.ndarray] = []
        # Sort frame ids so point order is deterministic across runs.
        for frame_id in sorted(tf.T_map_agents.keys()):
            if frame_id not in assets.dynamic_agent_assets:
                continue
            asset = assets.dynamic_agent_assets[frame_id]
            pts_map = transform_points(tf.T_map_agents[frame_id], asset.points)
            cols = asset.colors
            inten = asset.intensity
            if cull_radius_m is not None:
                d_xy = pts_map[:, :2] - ego_xy
                # Same squared-distance cull used for static subsets.
                keep = (d_xy[:, 0] * d_xy[:, 0] + d_xy[:, 1] * d_xy[:, 1]) <= (cull_radius_m * cull_radius_m)
                pts_map = pts_map[keep]
                cols = cols[keep]
                inten = inten[keep]
            if pts_map.shape[0] == 0:
                continue
            pts_list.append(pts_map)
            col_list.append(cols)
            int_list.append(inten)
        if pts_list:
            agent_points_map = np.concatenate(pts_list, axis=0)
            agent_colors = np.concatenate(col_list, axis=0)
            agent_intensity = np.concatenate(int_list, axis=0)
        else:
            agent_points_map, agent_colors, agent_intensity = _empty_agent_arrays()
    else:
        agent_points_map, agent_colors, agent_intensity = _empty_agent_arrays()

    return agent_points_map, agent_colors, agent_intensity


def compose_scene(
    t: float,
    assets: SceneAssets,
    tf: TFSnapshot,
    include_agent: bool = False,
    cull_radius_m: float = 110.0,
    cull_cache: SpatialCullCache | None = None,
) -> ComposedScene:
    """Build composed scene at time t.

    Spatially culls static geometry within cull_radius_m of ego using KD-tree.
    Only dynamic-agent points are transformed per frame.
    Set include_agent=True to include dynamic car geometry.
    """
    if cull_cache is None:
        raise ValueError("compose_scene requires a SpatialCullCache instance")

    ego_xy = tf.T_map_base[:2, 3]
    static_subsets = _compose_static_subsets(t, assets, ego_xy, cull_radius_m, cull_cache)

    agent_points_map, agent_colors, agent_intensity = _compose_dynamic_agents(
        assets, tf, ego_xy, include_agent, cull_radius_m
    )

    return ComposedScene(
        t=t,
        tf=tf,
        static_points=static_subsets.points,
        static_colors=static_subsets.colors,
        static_labels=static_subsets.labels,
        static_intensity=static_subsets.intensity,
        static_car_label_id=assets.label_car_id,
        static_cars_fullres_points=static_subsets.fullres_car_points,
        static_cars_fullres_colors=static_subsets.fullres_car_colors,
        agent_points_map=agent_points_map,
        agent_colors=agent_colors,
        agent_intensity=agent_intensity,
    )
