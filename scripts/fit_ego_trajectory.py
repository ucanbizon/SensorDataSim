"""Fit a more rigorous ego trajectory from processed scene labels.

This helper uses the processed static scene + semantic labels to:
1) estimate a road-aligned coordinate frame,
2) fit a road centerline from Ground/Road_markings,
3) score lane-offset candidates using static Car proximity and road support,
4) emit a proposed ego waypoint list for sim.yaml (stored as [Y, X] pairs).

It does NOT change trajectory generation math in sim/trajectory.py.

Usage examples:
    conda run -n sensorsim python scripts/fit_ego_trajectory.py
    conda run -n sensorsim python scripts/fit_ego_trajectory.py --write-config
    conda run -n sensorsim python scripts/fit_ego_trajectory.py --lane-offsets "-8,-6,-4,-2,2,4,6,8"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from scipy.spatial import cKDTree

# Ensure local imports work when run as a script from repo root or scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.assets import load_assets
from sim.config import SimConfig, load_config


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RoadFrame:
    origin_xy: np.ndarray  # (2,)
    u_hat: np.ndarray      # (2,) longitudinal
    v_hat: np.ndarray      # (2,) lateral (right-handed in XY plane)


@dataclass
class CenterlineModel:
    s_valid: np.ndarray
    d_center: np.ndarray
    d_low: np.ndarray
    d_high: np.ndarray
    support_count: np.ndarray
    s_bin_m: float


@dataclass
class CandidateMetrics:
    lane_offset_m: float
    direction_sign: int
    desired_direction: bool
    score: float
    car_clear_q10_m: float
    car_clear_median_m: float
    car_overlap_frac: float
    road_support_frac: float
    road_nearest_q90_m: float
    lane_violation_frac: float
    min_scene_edge_margin_m: float
    hard_valid: bool


# ---------------------------------------------------------------------------
# CLI and utility parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit ego lane waypoints from processed scene labels")
    parser.add_argument("--config", default="sim.yaml", help="Path to sim.yaml")
    parser.add_argument(
        "--lane-offsets",
        default="-8,-6,-4,-2,2,4,6,8",
        help="Comma-separated lane offset candidates (m) relative to fitted centerline",
    )
    parser.add_argument("--s-bin", type=float, default=2.0, help="Longitudinal bin size in meters")
    parser.add_argument("--waypoints", type=int, default=6, help="Number of output waypoints")
    parser.add_argument("--samples", type=int, default=300, help="Number of dense path samples for scoring")
    parser.add_argument("--road-min-count", type=int, default=200, help="Min road points per s-bin to treat centerline bin as valid")
    parser.set_defaults(prefer_opposite_current=True)
    parser.add_argument(
        "--no-prefer-opposite-current",
        dest="prefer_opposite_current",
        action="store_false",
        help="Disable opposite-direction preference when scoring candidates",
    )
    parser.add_argument("--write-config", action="store_true", help="Patch sim.yaml ego.waypoints in place (creates .bak)")
    return parser.parse_args()


def parse_offset_list(text: str) -> list[float]:
    """Parse comma-separated lane offsets in meters."""
    vals = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise ValueError("No lane offsets provided")
    return vals


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """Small edge-padded moving average used to stabilize centerline bins."""
    if k <= 1 or x.size == 0:
        return x.copy()
    k = int(k)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(xpad, kernel, mode="valid")


def fit_road_frame(road_xy: np.ndarray, current_wp_xy: np.ndarray) -> RoadFrame:
    """Fit a road-aligned 2D frame from road points using PCA."""
    # PCA gives the dominant road direction from road points.
    origin = np.median(road_xy, axis=0)
    X = road_xy - origin
    cov = (X.T @ X) / max(1, X.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    u = eigvecs[:, int(np.argmax(eigvals))]
    u = u / np.linalg.norm(u)
    v = np.array([-u[1], u[0]], dtype=np.float64)

    # Orient u so current ego waypoints progress in +s (for stable direction semantics)
    s_current, _ = project_sd(current_wp_xy, origin, u, v)
    if s_current[-1] < s_current[0]:
        u = -u
        v = -v

    return RoadFrame(origin_xy=origin, u_hat=u, v_hat=v)


def project_sd(xy: np.ndarray, origin: np.ndarray, u_hat: np.ndarray, v_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project XY map points into the fitted road frame (s, d)."""
    rel = xy - origin[None, :]
    s = rel @ u_hat
    d = rel @ v_hat
    return s, d


def unproject_sd(s: np.ndarray, d: np.ndarray, frame: RoadFrame) -> np.ndarray:
    """Map road-frame coordinates (s, d) back into XY map coordinates."""
    return frame.origin_xy[None, :] + s[:, None] * frame.u_hat[None, :] + d[:, None] * frame.v_hat[None, :]


def fit_centerline(road_s: np.ndarray, road_d: np.ndarray, s_bin_m: float, min_count: int) -> CenterlineModel:
    """Fit a median road centerline and road envelope in longitudinal bins."""
    s_min = np.floor(road_s.min() / s_bin_m) * s_bin_m
    s_max = np.ceil(road_s.max() / s_bin_m) * s_bin_m
    edges = np.arange(s_min, s_max + s_bin_m, s_bin_m, dtype=np.float64)
    if edges.size < 2:
        raise ValueError("Road longitudinal extent too small to bin")

    idx = np.digitize(road_s, edges) - 1
    n_bins = edges.size - 1
    s_centers = (edges[:-1] + edges[1:]) * 0.5

    d_center = np.full(n_bins, np.nan, dtype=np.float64)
    d_low = np.full(n_bins, np.nan, dtype=np.float64)
    d_high = np.full(n_bins, np.nan, dtype=np.float64)
    support = np.zeros(n_bins, dtype=np.int32)

    for b in range(n_bins):
        mask = (idx == b)
        cnt = int(np.count_nonzero(mask))
        support[b] = cnt
        if cnt < min_count:
            continue
        dvals = road_d[mask]
        d_center[b] = float(np.median(dvals))
        d_low[b] = float(np.quantile(dvals, 0.10))
        d_high[b] = float(np.quantile(dvals, 0.90))

    valid = np.isfinite(d_center)
    if np.count_nonzero(valid) < 5:
        raise ValueError("Too few valid centerline bins; road support insufficient")

    # Smooth only the bins that had enough road support.
    d_center_valid = moving_average(d_center[valid], 7)
    d_low_valid = moving_average(d_low[valid], 7)
    d_high_valid = moving_average(d_high[valid], 7)

    return CenterlineModel(
        s_valid=s_centers[valid],
        d_center=d_center_valid,
        d_low=d_low_valid,
        d_high=d_high_valid,
        support_count=support[valid],
        s_bin_m=float(s_bin_m),
    )


def interp_centerline(centerline: CenterlineModel, s_query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the fitted centerline and lane envelope at query s positions."""
    s_ref = centerline.s_valid
    d_center = np.interp(s_query, s_ref, centerline.d_center, left=np.nan, right=np.nan)
    d_low = np.interp(s_query, s_ref, centerline.d_low, left=np.nan, right=np.nan)
    d_high = np.interp(s_query, s_ref, centerline.d_high, left=np.nan, right=np.nan)
    return d_center, d_low, d_high


def compute_current_ego_xy(cfg: SimConfig) -> np.ndarray:
    # sim.yaml convention is [Y, X]
    return np.array([(wp[1], wp[0]) for wp in cfg.ego.waypoints], dtype=np.float64)


def build_candidate_path(
    s_start: float,
    s_end: float,
    n_samples: int,
    direction_sign: int,
    lane_offset_m: float,
    centerline: CenterlineModel,
    frame: RoadFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one lane candidate path in both geometry and road-frame coordinates."""
    s_asc = np.linspace(min(s_start, s_end), max(s_start, s_end), n_samples, dtype=np.float64)
    d_center, d_low, d_high = interp_centerline(centerline, s_asc)
    d_path = d_center + lane_offset_m
    xy_asc = unproject_sd(s_asc, d_path, frame)

    if direction_sign > 0:
        return s_asc, d_path, d_low, d_high, xy_asc
    return s_asc[::-1], d_path[::-1], d_low[::-1], d_high[::-1], xy_asc[::-1]


def nearest_dist_stats(tree: cKDTree | None, pts_xy: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Nearest-neighbor distances plus q10/median summary stats."""
    if tree is None or pts_xy.shape[0] == 0:
        d = np.full(pts_xy.shape[0], np.inf, dtype=np.float64)
        return d, float(np.inf), float(np.inf)
    dists, _ = tree.query(pts_xy, k=1)
    dists = np.asarray(dists, dtype=np.float64)
    return dists, float(np.quantile(dists, 0.10)), float(np.median(dists))


def score_candidates(
    lane_offsets: Iterable[float],
    road_frame: RoadFrame,
    centerline: CenterlineModel,
    current_wp_xy: np.ndarray,
    road_xy: np.ndarray,
    car_xy: np.ndarray,
    scene_min_xy: np.ndarray,
    scene_max_xy: np.ndarray,
    n_samples: int,
    prefer_opposite: bool,
) -> tuple[dict[tuple[float, int], dict], CandidateMetrics]:
    """Score lane-offset candidates against road support and car clearance.

    Returns a dict of detailed candidate paths keyed by (offset, direction_sign)
    and the best candidate metrics object.
    """
    current_s, _ = project_sd(current_wp_xy, road_frame.origin_xy, road_frame.u_hat, road_frame.v_hat)
    s0, s1 = float(current_s[0]), float(current_s[-1])
    current_dir_sign = 1 if (s1 - s0) >= 0 else -1
    desired_sign = -current_dir_sign if prefer_opposite else current_dir_sign

    road_tree = cKDTree(road_xy)
    car_tree = cKDTree(car_xy) if car_xy.shape[0] > 0 else None

    results: dict[tuple[float, int], dict] = {}
    best_metrics: CandidateMetrics | None = None

    lane_half_width_m = 1.75
    car_overlap_thresh_m = lane_half_width_m + 0.75
    road_support_thresh_m = 1.5
    lane_envelope_margin_m = 0.75

    # Try every lane offset in both travel directions, then score each path.
    # This intentionally keeps all candidates so other scripts (for example the
    # dynamic agent planner) can reuse the same scoring output.
    for offset in lane_offsets:
        for direction_sign in (+1, -1):
            s_path, d_path, d_low, d_high, xy_path = build_candidate_path(
                s0, s1, n_samples, direction_sign, float(offset), centerline, road_frame
            )

            finite_mask = np.isfinite(d_path) & np.isfinite(d_low) & np.isfinite(d_high)
            if not np.all(finite_mask):
                hard_valid = False
                car_q10 = car_median = np.inf
                car_overlap_frac = 1.0
                road_support_frac = 0.0
                road_q90 = np.inf
                lane_violation_frac = 1.0
                min_edge_margin = -np.inf
            else:
                # Road support: a good path stays close to road-labeled points.
                road_dists, _, road_median = nearest_dist_stats(road_tree, xy_path)
                road_support_frac = float(np.mean(road_dists <= road_support_thresh_m))
                road_q90 = float(np.quantile(road_dists, 0.90))

                # Car clearance: penalize paths that run through parked cars.
                car_dists, car_q10, car_median = nearest_dist_stats(car_tree, xy_path)
                car_overlap_frac = float(np.mean(car_dists <= car_overlap_thresh_m)) if np.isfinite(car_dists).any() else 0.0

                # Lane envelope validity: keep the path inside the observed road
                # support envelope with a small safety margin.
                below = d_path < (d_low - lane_envelope_margin_m)
                above = d_path > (d_high + lane_envelope_margin_m)
                lane_violation_frac = float(np.mean(below | above))

                # Scene edge margin prevents candidates that exit the processed tile.
                edge_dx = np.minimum(xy_path[:, 0] - scene_min_xy[0], scene_max_xy[0] - xy_path[:, 0])
                edge_dy = np.minimum(xy_path[:, 1] - scene_min_xy[1], scene_max_xy[1] - xy_path[:, 1])
                min_edge_margin = float(np.minimum(edge_dx, edge_dy).min())

                hard_valid = (
                    road_support_frac >= 0.70 and
                    lane_violation_frac <= 0.20 and
                    min_edge_margin >= 5.0
                )

            desired = (direction_sign == desired_sign)
            direction_bonus = 1.0 if desired else -1.0

            # Weighted score emphasizes drivable support and car-free space.
            score = (
                5.0 * road_support_frac
                - 8.0 * car_overlap_frac
                - 4.0 * lane_violation_frac
                + 0.5 * float(np.clip(car_q10 if np.isfinite(car_q10) else 20.0, 0.0, 10.0))
                + 0.2 * float(np.clip(min_edge_margin if np.isfinite(min_edge_margin) else 0.0, 0.0, 10.0))
                + 0.5 * direction_bonus
            )
            if not hard_valid:
                score -= 100.0

            metrics = CandidateMetrics(
                lane_offset_m=float(offset),
                direction_sign=int(direction_sign),
                desired_direction=bool(desired),
                score=float(score),
                car_clear_q10_m=float(car_q10 if np.isfinite(car_q10) else 999.0),
                car_clear_median_m=float(car_median if np.isfinite(car_median) else 999.0),
                car_overlap_frac=float(car_overlap_frac),
                road_support_frac=float(road_support_frac),
                road_nearest_q90_m=float(road_q90 if np.isfinite(road_q90) else 999.0),
                lane_violation_frac=float(lane_violation_frac),
                min_scene_edge_margin_m=float(min_edge_margin if np.isfinite(min_edge_margin) else -999.0),
                hard_valid=bool(hard_valid),
            )
            results[(float(offset), int(direction_sign))] = {
                "metrics": metrics,
                "s_path": s_path,
                "d_path": d_path,
                "xy_path": xy_path,
                "d_low": d_low,
                "d_high": d_high,
            }

            if best_metrics is None or metrics.score > best_metrics.score:
                best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("No valid lane candidates found")
    return results, best_metrics


def sample_waypoints(candidate: dict, frame: RoadFrame, n_waypoints: int) -> list[list[float]]:
    """Downsample a dense candidate path into sim.yaml waypoints [Y, X]."""
    s_path = candidate["s_path"]
    d_path = candidate["d_path"]
    idx = np.linspace(0, len(s_path) - 1, n_waypoints).round().astype(int)
    idx = np.unique(np.clip(idx, 0, len(s_path) - 1))
    s_wp = s_path[idx]
    d_wp = d_path[idx]
    xy_wp = unproject_sd(s_wp, d_wp, frame)
    # sim.yaml uses [Y, X]
    return [[float(y), float(x)] for x, y in xy_wp]


def maybe_patch_config(config_path: Path, waypoints_yx: list[list[float]]) -> None:
    """Patch ego.waypoints in sim.yaml (creates a one-time .bak backup)."""
    backup = config_path.with_suffix(config_path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw.setdefault("ego", {})
    raw["ego"]["waypoints"] = waypoints_yx
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    lane_offsets = parse_offset_list(args.lane_offsets)

    # ------------------------------------------------------------------
    # Phase 1: load processed scene data and fit a road-aligned frame
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    assets = load_assets(cfg.data_dir)

    labels = assets.static_labels
    class_map = assets.class_map
    label_ground = class_map["Ground"]
    label_markings = class_map["Road_markings"]
    label_car = class_map["Car"]

    pts_xy = assets.static_points[:, :2].astype(np.float64, copy=False)
    road_mask = (labels == label_ground) | (labels == label_markings)
    car_mask = (labels == label_car)
    road_xy = pts_xy[road_mask]
    car_xy = pts_xy[car_mask]
    current_wp_xy = compute_current_ego_xy(cfg)

    road_frame = fit_road_frame(road_xy, current_wp_xy)
    road_s, road_d = project_sd(road_xy, road_frame.origin_xy, road_frame.u_hat, road_frame.v_hat)
    centerline = fit_centerline(road_s, road_d, s_bin_m=args.s_bin, min_count=args.road_min_count)

    scene_min_xy = assets.static_points[:, :2].min(axis=0)
    scene_max_xy = assets.static_points[:, :2].max(axis=0)

    # ------------------------------------------------------------------
    # Phase 2: score lane candidates and choose the best one
    # ------------------------------------------------------------------
    results, best_metrics = score_candidates(
        lane_offsets=lane_offsets,
        road_frame=road_frame,
        centerline=centerline,
        current_wp_xy=current_wp_xy,
        road_xy=road_xy,
        car_xy=car_xy,
        scene_min_xy=scene_min_xy,
        scene_max_xy=scene_max_xy,
        n_samples=args.samples,
        prefer_opposite=args.prefer_opposite_current,
    )
    best_key = (best_metrics.lane_offset_m, best_metrics.direction_sign)
    best_candidate = results[best_key]
    waypoints_yx = sample_waypoints(best_candidate, road_frame, n_waypoints=args.waypoints)

    # ------------------------------------------------------------------
    # Phase 3: print YAML snippet and optionally patch sim.yaml
    # ------------------------------------------------------------------
    # Print a short summary and a YAML-ready block. The detailed metrics are still
    # available inside `best_metrics` if you want to inspect them in code.
    print("=" * 70)
    print("EGO TRAJECTORY FIT")
    print("=" * 70)
    print(f"Best lane offset: {best_metrics.lane_offset_m:+.2f} m")
    print(f"Best direction sign (road s): {best_metrics.direction_sign:+d}")
    print(f"Score: {best_metrics.score:.3f}")
    print("")
    print("Proposed sim.yaml snippet (ego.waypoints in [Y, X]):")
    print(yaml.safe_dump({"ego": {"waypoints": waypoints_yx}}, sort_keys=False))
    print("")

    if args.write_config:
        maybe_patch_config(Path(args.config), waypoints_yx)
        print(f"Patched {args.config} (backup created on first write: {args.config}.bak)")
    else:
        print("Config not modified (use --write-config to patch sim.yaml).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
