"""Numba-accelerated kernels for LiDAR and camera rendering."""

from __future__ import annotations

import math
import time

import numba as nb
import numpy as np


# ---------------------------------------------------------------------------
# LiDAR: fused transform → spherical → ring → tolerance → bin → scatter-min
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def lidar_scatter_min(
    points_map: np.ndarray,       # (N, 3) float64
    R_vel_map: np.ndarray,        # (3, 3) float64, contiguous
    t_vel_map: np.ndarray,        # (3,) float64
    min_range_sq: float,
    max_range_sq: float,
    elev_base_deg: float,         # -15.0
    ring_spacing_deg: float,      # 2.0
    tolerance_deg: float,         # 1.0
    num_rings: int,               # 16
    num_azimuth_bins: int,        # 1800
    azimuth_bin_width_rad: float,
    best_range: np.ndarray,       # (num_rings * num_azimuth_bins,) float64, pre-filled inf
    best_idx: np.ndarray,         # (num_rings * num_azimuth_bins,) int64, pre-filled -1
) -> tuple:  # (n_in_range, n_after_tol)
    """Single-pass LiDAR depth buffer: transform, gate, bin, scatter-min.

    Replaces the entire chain of ~25 intermediate NumPy arrays with one
    scalar loop.  No intermediate allocations.

    "scatter-min" means each point maps to one output cell and updates that
    cell only if its range is smaller than the current winner.
    """
    TWO_PI = 2.0 * math.pi
    RAD2DEG = 180.0 / math.pi
    n_in_range = 0
    n_after_tol = 0
    N = points_map.shape[0]

    for i in range(N):
        px = points_map[i, 0]
        py = points_map[i, 1]
        pz = points_map[i, 2]

        # Inline transform: p_vel = R @ p_map + t
        x = R_vel_map[0, 0] * px + R_vel_map[0, 1] * py + R_vel_map[0, 2] * pz + t_vel_map[0]
        y = R_vel_map[1, 0] * px + R_vel_map[1, 1] * py + R_vel_map[1, 2] * pz + t_vel_map[1]
        z = R_vel_map[2, 0] * px + R_vel_map[2, 1] * py + R_vel_map[2, 2] * pz + t_vel_map[2]

        # Range gate (squared — avoids sqrt for rejected points)
        # Squared gate is cheaper than sqrt for points that will be rejected.
        r_sq = x * x + y * y + z * z
        if r_sq < min_range_sq or r_sq > max_range_sq:
            continue
        n_in_range += 1

        rng = math.sqrt(r_sq)
        range_xy = math.sqrt(x * x + y * y)
        elev_deg = math.atan2(z, range_xy) * RAD2DEG

        # Nearest ring assignment
        ring_f = (elev_deg - elev_base_deg) / ring_spacing_deg
        ring = int(round(ring_f))
        if ring < 0 or ring >= num_rings:
            continue

        # Tolerance gate
        ring_center = elev_base_deg + ring * ring_spacing_deg
        if abs(elev_deg - ring_center) > tolerance_deg:
            continue
        n_after_tol += 1

        # Azimuth bin
        azim = math.atan2(y, x)
        azim_pos = azim % TWO_PI
        abin = int(azim_pos / azimuth_bin_width_rad)
        if abin >= num_azimuth_bins:
            abin = num_azimuth_bins - 1

        # Scatter-min: compare-and-swap
        # Store the original point index so the caller can gather winner
        # attributes after selection without duplicating arrays in this kernel.
        cell = ring * num_azimuth_bins + abin
        if rng < best_range[cell]:
            best_range[cell] = rng
            best_idx[cell] = i

    return n_in_range, n_after_tol


# ---------------------------------------------------------------------------
# Camera: projection + Gaussian depth-blended splatting
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def camera_project_frustum_pack(
    points_map: np.ndarray,      # (N,3) float64
    colors: np.ndarray,          # (N,3) uint8
    R_opt_map: np.ndarray,       # (3,3) float64, contiguous
    t_opt_map: np.ndarray,       # (3,) float64
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_range: float,
    max_range: float,
    W: int,
    H: int,
    margin_px: int,
    ui_out: np.ndarray,          # (N,) int32
    vi_out: np.ndarray,          # (N,) int32
    depths_out: np.ndarray,      # (N,) float64
    colors_out: np.ndarray,      # (N,3) uint8
) -> tuple:  # (n_in_front, n_in_frustum, n_packed)
    """Transform + range gate + project + frustum gate + pack valid points.

    This replaces several NumPy masking/compaction steps in camera._render_subset.
    Output arrays are preallocated by the caller; n_packed tells how much of the
    prefix is valid for this frame.
    """
    n_in_front = 0
    n_in_frustum = 0
    n_packed = 0
    N = points_map.shape[0]

    for i in range(N):
        px = points_map[i, 0]
        py = points_map[i, 1]
        pz = points_map[i, 2]

        x = R_opt_map[0, 0] * px + R_opt_map[0, 1] * py + R_opt_map[0, 2] * pz + t_opt_map[0]
        y = R_opt_map[1, 0] * px + R_opt_map[1, 1] * py + R_opt_map[1, 2] * pz + t_opt_map[1]
        z = R_opt_map[2, 0] * px + R_opt_map[2, 1] * py + R_opt_map[2, 2] * pz + t_opt_map[2]

        if z <= min_range or z >= max_range:
            continue
        n_in_front += 1

        inv_z = 1.0 / z
        u = fx * x * inv_z + cx
        v = fy * y * inv_z + cy

        # Round to nearest pixel center to match the Python reference path.
        ui = int(round(u))
        vi = int(round(v))

        if ui < -margin_px or ui >= (W + margin_px) or vi < -margin_px or vi >= (H + margin_px):
            continue
        n_in_frustum += 1

        ui_out[n_packed] = ui
        vi_out[n_packed] = vi
        depths_out[n_packed] = z
        colors_out[n_packed, 0] = colors[i, 0]
        colors_out[n_packed, 1] = colors[i, 1]
        colors_out[n_packed, 2] = colors[i, 2]
        n_packed += 1

    return n_in_front, n_in_frustum, n_packed


@nb.njit(cache=True)
def splat_gaussian_depth_front(
    ui: np.ndarray,            # (K,) int32, pixel x coords
    vi: np.ndarray,            # (K,) int32, pixel y coords
    depths: np.ndarray,        # (K,) float64
    radii: np.ndarray,         # (K,) int16, per-point splat radius
    W: int,
    H: int,
    depth_front: np.ndarray,   # (H, W) float64, pre-filled inf
) -> None:
    """Pass A of Gaussian splatting: write nearest depth per pixel only."""
    K = ui.shape[0]
    for i in range(K):
        cx = ui[i]
        cy = vi[i]
        d = depths[i]
        r = int(radii[i])
        r2 = r * r

        for dy in range(-r, r + 1):
            vy = cy + dy
            if vy < 0 or vy >= H:
                continue
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r2:
                    continue
                ux = cx + dx
                if ux < 0 or ux >= W:
                    continue
                if d < depth_front[vy, ux]:
                    depth_front[vy, ux] = d


@nb.njit(cache=True)
def splat_gaussian_accumulate(
    ui: np.ndarray,            # (K,) int32, pixel x coords
    vi: np.ndarray,            # (K,) int32, pixel y coords
    depths: np.ndarray,        # (K,) float64
    colors: np.ndarray,        # (K, 3) uint8
    radii: np.ndarray,         # (K,) int16, per-point splat radius
    W: int,
    H: int,
    depth_front: np.ndarray,   # (H, W) float64, front surface from pass A
    accum_rgb: np.ndarray,     # (H, W, 3) float64, pre-filled 0
    accum_w: np.ndarray,       # (H, W) float64, pre-filled 0
    depth_margin_frac: float,
    depth_margin_min: float,
) -> None:
    """Pass B of Gaussian splatting: accumulate color near the front surface."""
    K = ui.shape[0]
    for i in range(K):
        cx = ui[i]
        cy = vi[i]
        d = depths[i]
        r = int(radii[i])
        r2 = r * r
        # sigma ~= r/3 (sharper than r/2) to reduce blur after supersample + downsample
        inv_2sigma2 = 4.5 / max(r2, 1)
        cr = float(colors[i, 0])
        cg = float(colors[i, 1])
        cb = float(colors[i, 2])

        for dy in range(-r, r + 1):
            vy = cy + dy
            if vy < 0 or vy >= H:
                continue
            for dx in range(-r, r + 1):
                dist2 = dx * dx + dy * dy
                if dist2 > r2:
                    continue
                ux = cx + dx
                if ux < 0 or ux >= W:
                    continue
                front = depth_front[vy, ux]
                margin = front * depth_margin_frac
                if margin < depth_margin_min:
                    margin = depth_margin_min
                if d > front + margin:
                    continue
                w = math.exp(-dist2 * inv_2sigma2)
                accum_rgb[vy, ux, 0] += w * cr
                accum_rgb[vy, ux, 1] += w * cg
                accum_rgb[vy, ux, 2] += w * cb
                accum_w[vy, ux] += w


# ---------------------------------------------------------------------------
# Warmup: compile all kernels once with tiny dummy data
# ---------------------------------------------------------------------------

def warmup_numba() -> float:
    """Trigger JIT compilation for all kernels. Returns warmup time in seconds."""
    t0 = time.perf_counter()

    # LiDAR kernel
    dummy_pts = np.zeros((10, 3), dtype=np.float64)
    dummy_R = np.eye(3, dtype=np.float64)
    dummy_t = np.zeros(3, dtype=np.float64)
    n_cells = 16 * 1800
    best_range = np.full(n_cells, np.inf, dtype=np.float64)
    best_idx = np.full(n_cells, -1, dtype=np.int64)
    lidar_scatter_min(
        dummy_pts, dummy_R, dummy_t,
        1.0, 10000.0,
        -15.0, 2.0, 1.0, 16, 1800,
        np.deg2rad(0.2),
        best_range, best_idx,
    )

    # Camera kernel
    dummy_pts_cam = np.zeros((10, 3), dtype=np.float64)
    dummy_cols = np.zeros((10, 3), dtype=np.uint8)
    dummy_R_cam = np.eye(3, dtype=np.float64)
    dummy_t_cam = np.zeros(3, dtype=np.float64)
    dummy_ui_out = np.empty(10, dtype=np.int32)
    dummy_vi_out = np.empty(10, dtype=np.int32)
    dummy_depth_out = np.empty(10, dtype=np.float64)
    dummy_col_out = np.empty((10, 3), dtype=np.uint8)
    camera_project_frustum_pack(
        dummy_pts_cam, dummy_cols, dummy_R_cam, dummy_t_cam,
        100.0, 100.0, 2.0, 2.0, 0.1, 100.0, 4, 4, 2,
        dummy_ui_out, dummy_vi_out, dummy_depth_out, dummy_col_out,
    )

    dummy_ui = np.zeros(10, dtype=np.int32)
    dummy_vi = np.zeros(10, dtype=np.int32)
    dummy_d = np.ones(10, dtype=np.float64)
    dummy_c = np.zeros((10, 3), dtype=np.uint8)
    dummy_r = np.ones(10, dtype=np.int16)
    dummy_df = np.full((4, 4), np.inf, dtype=np.float64)
    dummy_ar = np.zeros((4, 4, 3), dtype=np.float64)
    dummy_aw = np.zeros((4, 4), dtype=np.float64)
    splat_gaussian_depth_front(dummy_ui, dummy_vi, dummy_d, dummy_r, 4, 4, dummy_df)
    splat_gaussian_accumulate(
        dummy_ui, dummy_vi, dummy_d, dummy_c, dummy_r,
        4, 4, dummy_df, dummy_ar, dummy_aw, 0.05, 0.15,
    )

    dt = time.perf_counter() - t0
    print(f"  Numba warmup: {dt:.1f}s")
    return dt
