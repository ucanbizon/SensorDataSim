"""Local road-surface height lookup for trajectory Z/pitch following."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from .assets import SceneAssets

# -- Tuning constants for robust height estimation --
_RADIUS_M = 2.0        # Query radius in XY (meters).
_K_NEIGHBORS = 48      # Max KD-tree neighbors to inspect inside the radius.
_MIN_NEIGHBORS = 12    # Fallback to plane if too few nearby road samples exist.
_MIN_INLIERS = 8       # Fallback to plane if robust clipping rejects too many points.
_MAD_SCALE = 3.0       # Keep points within this many MADs from local median Z.
_MIN_CLIP_M = 0.08     # Lower bound on clipping width to avoid over-rejecting.


def _plane_z(x: float, y: float, plane: np.ndarray) -> float:
    a, b, c, d = plane
    # Fallback when local neighborhood is sparse/noisy.
    return float(-(a * x + b * y + d) / c)


@dataclass
class RoadSurfaceModel:
    """Robust local road height estimator from road-labeled point samples."""

    z: np.ndarray                     # (N,) float64
    tree: cKDTree
    ground_plane: np.ndarray          # [a,b,c,d] fallback
    labels_used: tuple[str, ...]
    build_time_s: float
    n_points: int
    weight_sigma_m: float = 1.0

    # Expose for diagnostic printing
    radius_m: float = _RADIUS_M
    k_neighbors: int = _K_NEIGHBORS

    def __call__(self, x: float, y: float) -> float:
        return self.height(float(x), float(y))

    def height(self, x: float, y: float) -> float:
        """Estimate local road height z(x, y) with robust smoothing."""
        d, idx = self.tree.query([x, y], k=_K_NEIGHBORS,
                                distance_upper_bound=_RADIUS_M)
        d = np.atleast_1d(np.asarray(d, dtype=np.float64))
        idx = np.atleast_1d(np.asarray(idx))

        # cKDTree marks missing neighbors with inf distance and out-of-range
        # indices when distance_upper_bound is used.
        valid = np.isfinite(d) & (idx >= 0) & (idx < self.z.shape[0])
        if np.count_nonzero(valid) < _MIN_NEIGHBORS:
            return _plane_z(x, y, self.ground_plane)

        d = d[valid]
        z = self.z[idx[valid]]

        # Median + MAD rejects curbs, parked-car bottoms, and label noise more
        # reliably than a plain mean on raw road-labeled points.
        z_med = float(np.median(z))
        dev = np.abs(z - z_med)
        mad = float(np.median(dev))
        clip_m = max(_MIN_CLIP_M, _MAD_SCALE * 1.4826 * mad)
        inliers = dev <= clip_m
        if np.count_nonzero(inliers) < _MIN_INLIERS:
            return _plane_z(x, y, self.ground_plane)

        d_in = d[inliers]
        z_in = z[inliers]

        sigma = max(self.weight_sigma_m, 0.25)
        # Gaussian distance weights smooth frame-to-frame queries while keeping
        # the estimate local enough for grade changes.
        w = np.exp(-0.5 * (d_in / sigma) ** 2)
        w_sum = float(np.sum(w))
        if w_sum <= 1e-12:
            return z_med
        return float(np.sum(w * z_in) / w_sum)


def build_road_surface_model_from_assets(
    assets: "SceneAssets",
    labels: tuple[str, ...] = ("Ground", "Road_markings"),
) -> RoadSurfaceModel:
    """Build a local road-height model from road-labeled static scene points."""
    labels_used: list[str] = []
    mask = np.zeros(assets.static_labels.shape, dtype=bool)
    for name in labels:
        if name in assets.class_map:
            mask |= (assets.static_labels == int(assets.class_map[name]))
            labels_used.append(name)
    if not labels_used:
        raise KeyError(f"None of requested road labels found in class_map: {labels}")

    pts = np.asarray(assets.static_points[mask], dtype=np.float64)
    if pts.shape[0] == 0:
        raise RuntimeError(f"No points found for road labels: {labels_used}")

    xy = np.ascontiguousarray(pts[:, :2], dtype=np.float64)
    z = np.ascontiguousarray(pts[:, 2], dtype=np.float64)

    t0 = time.perf_counter()
    tree = cKDTree(xy)
    dt = time.perf_counter() - t0

    return RoadSurfaceModel(
        z=z,
        tree=tree,
        ground_plane=np.asarray(assets.ground_plane, dtype=np.float64),
        labels_used=tuple(labels_used),
        build_time_s=dt,
        n_points=z.shape[0],
        weight_sigma_m=max(_RADIUS_M * 0.5, 0.5),
    )
