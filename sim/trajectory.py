"""Ego and agent spline trajectories.

Waypoints are (Y, X) tuples - Y is the road direction, X is lateral.
Each trajectory is parameterized by arc length, then mapped to time at
constant speed.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from .transforms import pose_to_matrix


def ground_z(x: float, y: float, plane: np.ndarray) -> float:
    """Evaluate ground plane z at (x, y).  plane = [a, b, c, d]."""
    a, b, c, d = plane
    return float(-(a * x + b * y + d) / c)


def _build_spline(waypoints: List[Tuple[float, float]],
                  speed_mps: float,
                  ) -> Tuple[Callable, Callable, float]:
    """Build arc-length-parameterized cubic splines for X(t), Y(t).

    Args:
        waypoints: [(Y0, X0), (Y1, X1), ...] in map frame.
        speed_mps: constant travel speed.

    Returns:
        (x_of_t, y_of_t, max_t) - callables and maximum valid time.
    """
    ys = np.array([wp[0] for wp in waypoints], dtype=np.float64)
    xs = np.array([wp[1] for wp in waypoints], dtype=np.float64)

    # Cumulative arc length along the polyline
    diffs = np.column_stack([np.diff(xs), np.diff(ys)])
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.zeros(len(waypoints), dtype=np.float64)
    s[1:] = np.cumsum(seg_lengths)
    total_arc = s[-1]

    spline_x = CubicSpline(s, xs, bc_type="natural")
    spline_y = CubicSpline(s, ys, bc_type="natural")
    max_t = total_arc / speed_mps

    def x_of_t(t: float) -> float:
        return float(spline_x(np.clip(speed_mps * t, 0, total_arc)))

    def y_of_t(t: float) -> float:
        return float(spline_y(np.clip(speed_mps * t, 0, total_arc)))

    return x_of_t, y_of_t, float(max_t)


def _heading_from_spline(x_of_t: Callable, y_of_t: Callable,
                         t: float, dt: float = 1e-4) -> float:
    """Compute yaw from trajectory tangent via centered finite difference."""
    t1 = max(t - dt / 2, 0.0)
    t2 = t + dt / 2
    dx = x_of_t(t2) - x_of_t(t1)
    dy = y_of_t(t2) - y_of_t(t1)
    return float(np.arctan2(dy, dx))


def _moving_average_edge(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding (no zero-padding artifacts)."""
    if values.size <= 2:
        return values.astype(np.float64, copy=True)

    # Clamp to a valid odd window so the filter stays centered.
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    max_odd = values.size if values.size % 2 == 1 else values.size - 1
    window = min(window, max_odd)
    if window <= 1:
        return values.astype(np.float64, copy=True)

    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values.astype(np.float64, copy=False), (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Z-profile: pre-sampled and smoothed road height along the trajectory
# ---------------------------------------------------------------------------

def _build_z_profile(x_of_t, y_of_t, z_eval, speed_mps, total_arc,
                     sample_ds_m=0.25, smooth_window_m=16.0):
    """Sample z along the trajectory and smooth to suppress LiDAR road noise.

    Returns (profile_s, profile_z) arrays for use with np.interp,
    or (None, None) if the trajectory is too short to sample.
    """
    if total_arc <= 0.0 or speed_mps <= 0.0:
        return None, None

    n_samp = max(2, int(np.ceil(total_arc / sample_ds_m)) + 1)
    profile_s = np.linspace(0.0, total_arc, n_samp, dtype=np.float64)
    t_samples = profile_s / float(speed_mps)

    z_raw = np.empty(n_samp, dtype=np.float64)
    for i, t_s in enumerate(t_samples):
        z_raw[i] = float(z_eval(x_of_t(float(t_s)), y_of_t(float(t_s))))

    ds_eff = total_arc / max(n_samp - 1, 1)
    win = int(round(smooth_window_m / max(ds_eff, 1e-6)))
    if win % 2 == 0:
        win += 1
    profile_z = _moving_average_edge(z_raw, win)

    # Preserve exact start/end heights.
    profile_z[0] = z_raw[0]
    profile_z[-1] = z_raw[-1]
    return profile_s, profile_z


def _pitch_from_grade(s_t, yaw, x, y, z_eval, profile_s, profile_z,
                      total_arc, baseline_m, max_pitch_rad):
    """Estimate body pitch from local road grade along the heading direction.

    Uses the smoothed z-profile if available, otherwise queries z_eval directly.
    Uphill nose-up is negative pitch in map/body convention.
    """
    half = 0.5 * baseline_m
    if profile_s is not None and profile_z is not None:
        s1 = max(0.0, s_t - half)
        s2 = min(total_arc, s_t + half)
        ds = max(s2 - s1, 1e-6)
        z_f = float(np.interp(s2, profile_s, profile_z))
        z_b = float(np.interp(s1, profile_s, profile_z))
        pitch = -float(np.arctan2(z_f - z_b, ds))
    else:
        dx = float(np.cos(yaw)) * half
        dy = float(np.sin(yaw)) * half
        z_f = float(z_eval(x + dx, y + dy))
        z_b = float(z_eval(x - dx, y - dy))
        pitch = -float(np.arctan2(z_f - z_b, 2.0 * half))

    return float(np.clip(pitch, -max_pitch_rad, max_pitch_rad))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_trajectory(waypoints: List[Tuple[float, float]],
                     speed_mps: float,
                     ground_plane: np.ndarray,
                     *,
                     z_fn: Callable[[float, float], float] | None = None,
                     pitch_from_grade: bool = True,
                     pitch_baseline_m: float = 6.0,
                     max_abs_pitch_deg: float = 12.0,
                     ) -> Tuple[Callable[[float], np.ndarray], float]:
    """Build a trajectory function: t -> 4x4 SE3 (T_map_body).

    Args:
        waypoints: [(Y, X), ...] in map frame.
        speed_mps: constant speed along the path.
        ground_plane: [a, b, c, d] for Z evaluation.
        z_fn: optional local road-height function z(x, y). If omitted, uses
            the global ground plane.
        pitch_from_grade: estimate pitch from local road grade.
        pitch_baseline_m: central-difference baseline for grade estimate.
        max_abs_pitch_deg: clamp pitch magnitude for stability.

    Returns:
        (pose_fn, max_t) where pose_fn(t) -> 4x4 T_map_body.
    """
    x_of_t, y_of_t, max_t = _build_spline(waypoints, speed_mps)
    z_eval = z_fn if z_fn is not None else (lambda x, y: ground_z(x, y, ground_plane))
    max_pitch_rad = np.deg2rad(float(max_abs_pitch_deg))
    total_arc = float(max_t * speed_mps)

    # Pre-sample and smooth z along the path to suppress LiDAR road noise.
    profile_s, profile_z = (
        _build_z_profile(x_of_t, y_of_t, z_eval, speed_mps, total_arc)
        if z_fn is not None else (None, None)
    )

    def pose_fn(t: float) -> np.ndarray:
        t_c = float(np.clip(t, 0.0, max_t))
        s_t = float(np.clip(speed_mps * t_c, 0.0, total_arc))
        x = x_of_t(t_c)
        y = y_of_t(t_c)

        if profile_s is not None:
            z = float(np.interp(s_t, profile_s, profile_z))
        else:
            z = float(z_eval(x, y))

        yaw = _heading_from_spline(x_of_t, y_of_t, t_c)

        pitch = 0.0
        if z_fn is not None and pitch_from_grade and pitch_baseline_m > 0.0:
            pitch = _pitch_from_grade(
                s_t, yaw, x, y, z_eval, profile_s, profile_z,
                total_arc, pitch_baseline_m, max_pitch_rad,
            )

        return pose_to_matrix(x, y, z, 0.0, pitch, yaw)

    return pose_fn, max_t
