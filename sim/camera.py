"""Pinhole camera renderer: Gaussian splatting with depth-aware blending.

Camera optical frame convention (ROS REP-103):
    X = right, Y = down, Z = forward
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import CameraIntrinsics, CameraRenderConfig
from .numba_kernels import (
    camera_project_frustum_pack,
    splat_gaussian_accumulate,
    splat_gaussian_depth_front,
)
from .scene_compose import ComposedScene
from .transforms import invert_se3

# -- Tuning constants (never mutated at runtime) --
# Adaptive splat radius
_MIN_RADIUS_PX = 1
_MAX_RADIUS_PX = 6
_MAX_EXTRA_RADIUS_PX = 4
_KNEE_M = 10.0           # depth where radius starts shrinking
_KNEE_POWER = 2.5
# Density-based radius boost
_DENSITY_TILE_PX = 16
_DENSITY_REF_QUANTILE = 0.35
_DENSITY_GAIN_BETA = 0.55
_DENSITY_GAIN_MAX = 1.5
_DENSITY_DEPTH_FADE_M = 25.0
_DENSITY_MAX_EXTRA_PX = 1
# Gaussian depth blending
_DEPTH_MARGIN_FRAC = 0.05
_DEPTH_MARGIN_MIN_M = 0.15

# -- Reusable scratch buffers (single-threaded, allocated on first use) --
_pack_bufs: dict | None = None
_frame_bg: np.ndarray | None = None
_frame_depth_front: np.ndarray | None = None
_frame_accum_rgb: np.ndarray | None = None
_frame_accum_w: np.ndarray | None = None


@dataclass
class CameraFrame:
    image: np.ndarray    # (H, W, 3) uint8, RGB
    width: int
    height: int


@dataclass
class CameraStats:
    n_total: int
    n_in_front: int
    n_in_frustum: int
    n_pixels_filled: int
    fill_fraction: float
    depth_min: float
    depth_max: float


@dataclass
class ProjectedSubset:
    """Projected points ready for Gaussian splatting."""

    ui: np.ndarray       # (K,) int32 pixel x
    vi: np.ndarray       # (K,) int32 pixel y
    depths: np.ndarray   # (K,) float64 depth in camera optical frame
    colors: np.ndarray   # (K, 3) uint8 RGB
    radii: np.ndarray    # (K,) int16 splat radius in pixels


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------

def _make_background(H: int, W: int) -> np.ndarray:
    """Sky/ground gradient so missing pixels are not pure black."""
    top = np.array([14, 18, 28], dtype=np.float32)
    horizon = np.array([58, 62, 72], dtype=np.float32)
    bottom = np.array([26, 26, 28], dtype=np.float32)
    rows = np.zeros((H, 3), dtype=np.float32)
    split = max(1, min(H - 1, int(round(0.56 * H))))
    for c in range(3):
        rows[:split, c] = np.linspace(top[c], horizon[c], split)
        rows[split:, c] = np.linspace(horizon[c], bottom[c], H - split)
    return np.repeat(rows[:, None, :], W, axis=1).astype(np.uint8)


# ---------------------------------------------------------------------------
# Scratch buffer management
# ---------------------------------------------------------------------------

def _get_pack_bufs(n: int) -> dict:
    """Reusable projection output buffers, grown as needed."""
    global _pack_bufs
    if _pack_bufs is not None and _pack_bufs["capacity"] >= n:
        return _pack_bufs
    cap = max(n, 1024)
    _pack_bufs = {
        "capacity": cap,
        "ui": np.empty(cap, dtype=np.int32),
        "vi": np.empty(cap, dtype=np.int32),
        "depths": np.empty(cap, dtype=np.float64),
        "colors": np.empty((cap, 3), dtype=np.uint8),
    }
    return _pack_bufs


def _get_frame_bufs(H_int: int, W_int: int, H_out: int, W_out: int):
    """Reusable accumulation + background buffers."""
    global _frame_bg, _frame_depth_front, _frame_accum_rgb, _frame_accum_w
    if _frame_bg is None or _frame_bg.shape[:2] != (H_out, W_out):
        _frame_bg = _make_background(H_out, W_out)
    if _frame_depth_front is None or _frame_depth_front.shape != (H_int, W_int):
        _frame_depth_front = np.empty((H_int, W_int), dtype=np.float64)
        _frame_accum_rgb = np.empty((H_int, W_int, 3), dtype=np.float64)
        _frame_accum_w = np.empty((H_int, W_int), dtype=np.float64)
    return _frame_bg, _frame_depth_front, _frame_accum_rgb, _frame_accum_w


# ---------------------------------------------------------------------------
# Splat radius computation
# ---------------------------------------------------------------------------

def _adaptive_radii(depths: np.ndarray, base_radius: int, ss: int = 1,
                    max_radius_px: int | None = None) -> np.ndarray:
    """Depth-adaptive splat radii: larger up close, smaller at distance.

    ss (supersample factor) scales all pixel-unit limits proportionally.
    """
    if depths.size == 0:
        return np.empty(0, dtype=np.int16)
    base_r = max(1, base_radius)
    r_min = max(_MIN_RADIUS_PX * ss, base_r - 1)
    r_max = min(_MAX_RADIUS_PX * ss, base_r + _MAX_EXTRA_RADIUS_PX * ss)
    if max_radius_px is not None:
        r_max = min(r_max, max(1, max_radius_px))
    if r_max <= r_min:
        return np.full(depths.shape, r_min, dtype=np.int16)
    z = np.maximum(depths, 1e-3)
    # Smooth sigmoid-like falloff:
    # - near points stay close to r_max
    # - far points shrink toward r_min
    radii = r_min + (r_max - r_min) / (1.0 + np.power(z / _KNEE_M, _KNEE_POWER))
    return np.clip(np.rint(radii), r_min, r_max).astype(np.int16)


def _density_radius_gain(ui: np.ndarray, vi: np.ndarray,
                         depths: np.ndarray, W: int, H: int,
                         ss: int = 1) -> np.ndarray:
    """Per-point radius multiplier based on coarse tile density.

    Boosts splat radius in sparse screen regions to reduce holes.
    """
    n = depths.size
    gains = np.ones(n, dtype=np.float32)
    if n == 0:
        return gains
    valid = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if not np.any(valid):
        return gains

    tile = max(4, _DENSITY_TILE_PX * ss)
    gw = (W + tile - 1) // tile
    gh = (H + tile - 1) // tile
    tx = (ui[valid] // tile).astype(np.int32)
    ty = (vi[valid] // tile).astype(np.int32)
    counts = np.bincount(ty * gw + tx, minlength=gh * gw).reshape(gh, gw).astype(np.int32)

    # 3x3 box filter smoothing
    p = np.pad(counts, 1, mode="constant")
    smooth = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
              p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
              p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]).astype(np.float64)
    nz = smooth[smooth > 0]
    if nz.size == 0:
        return gains

    ref = max(1.0, float(np.quantile(nz, _DENSITY_REF_QUANTILE)))
    tile_gain = np.clip(np.power(ref / np.maximum(smooth, 1.0), _DENSITY_GAIN_BETA),
                        1.0, _DENSITY_GAIN_MAX)
    g = tile_gain[ty, tx].astype(np.float32)
    fade = 1.0 / (1.0 + (depths[valid].astype(np.float64) / _DENSITY_DEPTH_FADE_M) ** 2)
    gains[valid] = 1.0 + (g - 1.0) * fade.astype(np.float32)
    return gains


# ---------------------------------------------------------------------------
# Projection + splatting (two separate passes)
# ---------------------------------------------------------------------------

def _project_subset(points_map, colors, T_opt_map, intrinsics,
                    min_range, max_range, splat_radius, ss,
                    max_radius_px=None, use_density_gain=False):
    """Project points into camera pixels and compute per-point splat radii.

    Returns (projected, stats) where projected is a ProjectedSubset or None if
    nothing survived frustum culling.
    Stats is always (n_total, n_in_front, n_in_frustum).
    """
    n_total = points_map.shape[0]
    if n_total == 0:
        return None, (0, 0, 0)

    W, H = intrinsics.width, intrinsics.height
    cap = max(1, max_radius_px) if max_radius_px is not None else min(
        _MAX_RADIUS_PX * ss, max(1, splat_radius) + _MAX_EXTRA_RADIUS_PX * ss
    )
    density_extra = _DENSITY_MAX_EXTRA_PX * ss if use_density_gain else 0
    margin = cap + density_extra

    R = np.ascontiguousarray(T_opt_map[:3, :3])
    t = np.ascontiguousarray(T_opt_map[:3, 3])
    cols_in = colors if (colors.dtype == np.uint8 and colors.flags.c_contiguous) \
        else np.ascontiguousarray(colors)

    buf = _get_pack_bufs(n_total)
    n_in_front, n_in_frustum, n_packed = camera_project_frustum_pack(
        points_map, cols_in, R, t,
        float(intrinsics.fx), float(intrinsics.fy),
        float(intrinsics.cx), float(intrinsics.cy),
        float(min_range), float(max_range),
        W, H, margin,
        buf["ui"], buf["vi"], buf["depths"], buf["colors"],
    )
    if n_packed == 0:
        return None, (n_total, int(n_in_front), 0)

    # Copy from shared buffer - projections must persist across both passes.
    ui = buf["ui"][:n_packed].copy()
    vi = buf["vi"][:n_packed].copy()
    depths = buf["depths"][:n_packed].copy()
    cols = buf["colors"][:n_packed].copy()

    radii = _adaptive_radii(depths, splat_radius, ss, max_radius_px)
    if use_density_gain:
        gain = _density_radius_gain(ui, vi, depths, W, H, ss)
        radii = np.clip(np.rint(radii * gain), _MIN_RADIUS_PX * ss,
                        cap + density_extra).astype(np.int16)

    return (
        ProjectedSubset(ui=ui, vi=vi, depths=depths, colors=cols, radii=radii),
        (n_total, int(n_in_front), int(n_in_frustum)),
    )


def _write_depth_front(proj: ProjectedSubset, depth_front: np.ndarray, W: int, H: int) -> None:
    """Pass A: write nearest depth per pixel from projected points."""
    splat_gaussian_depth_front(proj.ui, proj.vi, proj.depths, proj.radii, W, H, depth_front)


def _accumulate_color(
    proj: ProjectedSubset,
    depth_front: np.ndarray,
    accum_rgb: np.ndarray,
    accum_w: np.ndarray,
    W: int,
    H: int,
) -> None:
    """Pass B: accumulate Gaussian-weighted color near the front surface."""
    splat_gaussian_accumulate(
        proj.ui, proj.vi, proj.depths, np.ascontiguousarray(proj.colors), proj.radii,
        W, H, depth_front, accum_rgb, accum_w,
        _DEPTH_MARGIN_FRAC, _DEPTH_MARGIN_MIN_M,
    )


# ---------------------------------------------------------------------------
# Main render entry point
# ---------------------------------------------------------------------------

def render_camera_frame(
    scene: ComposedScene,
    intrinsics: CameraIntrinsics,
    camera_cfg: CameraRenderConfig,
) -> tuple[CameraFrame, CameraStats]:
    """Render one camera image from a composed scene.

    Two-pass Gaussian splatting:
      Pass A - write nearest depth per pixel (all subsets)
      Pass B - accumulate Gaussian-weighted color near the front surface
    If supersample > 1, renders at higher internal resolution then downsamples.
    """
    tf = scene.tf
    W_out, H_out = intrinsics.width, intrinsics.height
    ss = camera_cfg.supersample
    W_int, H_int = W_out * ss, H_out * ss
    min_r = camera_cfg.min_range_m
    max_r = camera_cfg.max_range_m
    T_opt_map = invert_se3(tf.T_map_camera_optical)

    # Scale intrinsics for supersampled internal rendering
    intr_int = (
        CameraIntrinsics(
            width=W_int, height=H_int,
            fx=intrinsics.fx * ss, fy=intrinsics.fy * ss,
            cx=intrinsics.cx * ss, cy=intrinsics.cy * ss,
        )
        if ss > 1 else intrinsics
    )
    splat_r = camera_cfg.splat_radius_px * ss
    car_splat_r = camera_cfg.car_splat_radius_px * ss
    car_cap = car_splat_r + 2 * ss

    # -- Prepare buffers --
    bg, depth_front, accum_rgb, accum_w = _get_frame_bufs(H_int, W_int, H_out, W_out)
    depth_front.fill(np.inf)
    accum_rgb.fill(0.0)
    accum_w.fill(0.0)

    # -- Build point subsets --
    # Exclude labeled cars from the base pass - they are rendered separately
    # at full resolution for sharper car geometry.
    pts_base = scene.static_points
    cols_base = scene.static_colors
    if scene.static_car_label_id is not None:
        keep = scene.static_labels != scene.static_car_label_id
        pts_base = pts_base[keep]
        cols_base = cols_base[keep]

    has_fullres = (scene.static_cars_fullres_points is not None
                   and scene.static_cars_fullres_colors is not None)
    has_agents = scene.agent_points_map.shape[0] > 0

    # Each subset: (points, colors, splat_radius, max_radius_px, use_density_gain)
    subsets = [(pts_base, cols_base, splat_r, None, True)]
    if has_fullres:
        subsets.append((scene.static_cars_fullres_points,
                        scene.static_cars_fullres_colors,
                        car_splat_r, car_cap, False))
    if has_agents:
        subsets.append((scene.agent_points_map, scene.agent_colors,
                        car_splat_r, car_cap, False))

    # -- Project all subsets once --
    projections = []
    n_total = n_front = n_frustum = 0
    for pts, cols, r, cap, density in subsets:
        proj, (nt, nf, ni) = _project_subset(
            pts, cols, T_opt_map, intr_int, min_r, max_r, r, ss,
            max_radius_px=cap, use_density_gain=density,
        )
        n_total += nt
        n_front += nf
        n_frustum += ni
        if proj is not None:
            projections.append(proj)

    if n_front == 0:
        return (CameraFrame(bg.copy(), W_out, H_out),
                CameraStats(n_total, 0, 0, 0, 0.0, 0.0, 0.0))

    # -- Pass A: front depth from all subsets --
    for proj in projections:
        _write_depth_front(proj, depth_front, W_int, H_int)

    # -- Pass B: color accumulation against shared front depth --
    for proj in projections:
        _accumulate_color(proj, depth_front, accum_rgb, accum_w, W_int, H_int)

    # -- Normalize and compose output --
    filled = accum_w > 1e-6
    image_int = np.zeros((H_int, W_int, 3), dtype=np.uint8)
    inv_w = np.zeros_like(accum_w)
    inv_w[filled] = 1.0 / accum_w[filled]
    for c in range(3):
        image_int[:, :, c] = np.clip(accum_rgb[:, :, c] * inv_w, 0, 255).astype(np.uint8)

    if ss > 1:
        bg_int = cv2.resize(bg, (W_int, H_int), interpolation=cv2.INTER_LINEAR)
        image_int[~filled] = bg_int[~filled]
        image_out = cv2.resize(image_int, (W_out, H_out), interpolation=cv2.INTER_AREA)
    else:
        image_int[~filled] = bg[~filled]
        image_out = image_int

    # -- Stats --
    valid_d = depth_front[depth_front < np.inf]
    n_filled = int(np.count_nonzero(filled))
    fill_frac = n_filled / (W_int * H_int)
    return (
        CameraFrame(image=image_out, width=W_out, height=H_out),
        CameraStats(
            n_total=n_total, n_in_front=n_front, n_in_frustum=n_frustum,
            n_pixels_filled=n_filled, fill_fraction=fill_frac,
            depth_min=float(valid_d.min()) if valid_d.size else 0.0,
            depth_max=float(valid_d.max()) if valid_d.size else 0.0,
        ),
    )
