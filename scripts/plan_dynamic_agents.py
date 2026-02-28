"""Plan two dynamic agents and optionally patch sim.yaml.

This is a lightweight helper that reuses the lane-fitting logic from
`scripts/fit_ego_trajectory.py` to pick:
- one oncoming lane candidate
- one same-direction passing lane candidate

It then chooses two exported car assets and prints (or writes) the `agents:`
block expected by `sim.yaml`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# fit_ego_trajectory lives in scripts/ alongside this file; add to path so
# it can be imported as a module without packaging scripts/ as a package.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from sim.assets import load_assets
from sim.config import load_config

from fit_ego_trajectory import (
    compute_current_ego_xy,
    fit_centerline,
    fit_road_frame,
    parse_offset_list,
    project_sd,
    sample_waypoints,
    score_candidates,
)


# ---------------------------------------------------------------------------
# CLI and small helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plan two dynamic agents and patch sim.yaml")
    p.add_argument("--config", default="sim.yaml")
    p.add_argument("--asset-catalog", default="data/processed/agent_assets/asset_catalog.json")
    p.add_argument("--lane-offsets", default="-8,-6,-4,-2,2,4,6,8")
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--s-bin", type=float, default=2.0)
    p.add_argument("--road-min-count", type=int, default=200)
    p.add_argument("--waypoints", type=int, default=6)
    p.add_argument("--oncoming-offset", type=float, default=-8.0)
    p.add_argument("--passer-offset", type=float, default=2.0)
    p.add_argument("--oncoming-speed", type=float, default=9.0)
    p.add_argument("--passer-speed", type=float, default=13.0)
    p.add_argument("--oncoming-start-delay", type=float, default=0.5)
    p.add_argument("--passer-start-delay", type=float, default=3.0)
    p.add_argument("--write-config", action="store_true")
    return p.parse_args()


def _load_asset_catalog(path: Path) -> dict[str, Any]:
    # Keep this simple: if the file is missing, the exception is clear enough.
    return json.loads(path.read_text(encoding="utf-8"))


def _choose_agent_assets(catalog: dict[str, Any]) -> tuple[str, str]:
    """Pick two usable dynamic car assets from the exported catalog.

    I prefer moderate-size, plausible assets with decent side/end coverage, then
    keep the selection deterministic so repeated runs generate the same plan.
    """
    selected = catalog.get("selected_assets", [])

    # Build a quick lookup from exported asset-name prefix -> original cluster metrics.
    by_prefix: dict[str, dict[str, Any]] = {}
    for tile_entries in catalog.get("clusters_by_tile", {}).values():
        for c in tile_entries:
            prefix = f"{c['tile'].lower()}_car_cluster{int(c['cluster_id']):03d}_q"
            by_prefix[prefix] = c

    ranked: list[tuple[float, str, str]] = []  # (rank, asset_name, tile)
    for item in selected:
        name = item["asset_name"]
        src = next((v for k, v in by_prefix.items() if name.startswith(k)), None)
        if src is None:
            continue

        n_points = int(src["n_points"])
        m = src["metrics"]
        if n_points > 300_000 or m.get("roof_only_suspect") or not m.get("plausible_dims"):
            continue

        rank = (
            float(item.get("quality_score", 0.0))
            + float(m.get("side_frac", 0.0))
            + 0.5 * float(m.get("end_frac", 0.0))
            - 0.000005 * max(n_points - 120_000, 0)
        )
        ranked.append((rank, name, str(src.get("tile", ""))))

    ranked.sort(key=lambda x: x[0], reverse=True)
    if len(ranked) < 2:
        raise RuntimeError("Need at least two usable assets in asset_catalog.json")

    first_name, first_tile = ranked[0][1], ranked[0][2]
    second_name = next((name for _, name, tile in ranked[1:] if tile != first_tile), ranked[1][1])
    return first_name, second_name


def _build_lane_candidates(cfg, assets, lane_offsets, s_bin, road_min_count, n_samples):
    """Build lane candidates in a road-aligned coordinate frame.

    This mirrors the ego-fitting script but returns the full candidate set so I can
    pick different offsets/signs for different agent roles.
    """
    labels = assets.static_labels
    class_map = assets.class_map
    pts_xy = assets.static_points[:, :2].astype(np.float64, copy=False)

    road_mask = (labels == class_map["Ground"]) | (labels == class_map["Road_markings"])
    car_mask = labels == class_map["Car"]
    road_xy = pts_xy[road_mask]
    car_xy = pts_xy[car_mask]
    ego_wp_xy = compute_current_ego_xy(cfg)

    road_frame = fit_road_frame(road_xy, ego_wp_xy)
    road_s, road_d = project_sd(road_xy, road_frame.origin_xy, road_frame.u_hat, road_frame.v_hat)
    centerline = fit_centerline(road_s, road_d, s_bin_m=s_bin, min_count=road_min_count)

    scene_min_xy = assets.static_points[:, :2].min(axis=0)
    scene_max_xy = assets.static_points[:, :2].max(axis=0)
    results, _best = score_candidates(
        lane_offsets=lane_offsets,
        road_frame=road_frame,
        centerline=centerline,
        current_wp_xy=ego_wp_xy,
        road_xy=road_xy,
        car_xy=car_xy,
        scene_min_xy=scene_min_xy,
        scene_max_xy=scene_max_xy,
        n_samples=n_samples,
        prefer_opposite=False,  # I need both traffic directions available.
    )
    return road_frame, results


def _ego_direction_sign(cfg, road_frame) -> int:
    # Infer ego direction in the fitted road frame (+s or -s).
    wp_xy = compute_current_ego_xy(cfg)
    s, _ = project_sd(wp_xy, road_frame.origin_xy, road_frame.u_hat, road_frame.v_hat)
    return 1 if (s[-1] - s[0]) >= 0 else -1


def _candidate_waypoints(results: dict, road_frame, key: tuple[float, int], n_waypoints: int):
    """Fetch a scored candidate and sample sim.yaml-compatible waypoints."""
    candidate = results[key]
    return candidate, sample_waypoints(candidate, road_frame, n_waypoints=n_waypoints)


def _patch_sim_yaml_agents(config_path: Path, agents_block: dict[str, Any]) -> None:
    # Keep a one-time backup, then overwrite only the `agents:` section.
    backup = config_path.with_suffix(config_path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["agents"] = agents_block
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = parse_args()

    # ------------------------------------------------------------------
    # Phase 1: build lane candidates in the fitted road frame
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    assets = load_assets(cfg.data_dir)
    lane_offsets = parse_offset_list(args.lane_offsets)
    road_frame, results = _build_lane_candidates(
        cfg, assets, lane_offsets, args.s_bin, args.road_min_count, args.samples
    )

    # ------------------------------------------------------------------
    # Phase 2: choose lane candidates for each role
    # ------------------------------------------------------------------
    # Choose one oncoming lane and one same-direction lane.
    ego_sign = _ego_direction_sign(cfg, road_frame)
    oncoming_key = (float(args.oncoming_offset), -int(ego_sign))
    passer_key = (float(args.passer_offset), int(ego_sign))
    oncoming_cand, oncoming_wps = _candidate_waypoints(results, road_frame, oncoming_key, args.waypoints)
    passer_cand, passer_wps = _candidate_waypoints(results, road_frame, passer_key, args.waypoints)

    # ------------------------------------------------------------------
    # Phase 3: choose exported car assets and build the sim.yaml block
    # ------------------------------------------------------------------
    # Pick assets and build the exact block consumed by `sim.yaml`.
    oncoming_asset, passer_asset = _choose_agent_assets(_load_asset_catalog(Path(args.asset_catalog)))
    agents_block = {
        "oncoming": {
            "enabled": True,
            "frame_id": "agent_oncoming",
            "asset_name": oncoming_asset,
            "speed_mps": float(args.oncoming_speed),
            "start_delay_s": float(args.oncoming_start_delay),
            "waypoints": oncoming_wps,
        },
        "passer": {
            "enabled": True,
            "frame_id": "agent_passer",
            "asset_name": passer_asset,
            "speed_mps": float(args.passer_speed),
            "start_delay_s": float(args.passer_start_delay),
            "waypoints": passer_wps,
        },
    }

    # ------------------------------------------------------------------
    # Phase 4: print plan and optionally patch sim.yaml
    # ------------------------------------------------------------------
    print("=" * 70)
    print("DYNAMIC AGENT PLAN")
    print("=" * 70)
    print(f"Ego direction sign: {ego_sign:+d}")
    print(f"Oncoming candidate key: {oncoming_key}, score={oncoming_cand['metrics'].score:.3f}")
    print(f"Passer candidate key:   {passer_key}, score={passer_cand['metrics'].score:.3f}")
    print(f"Assets: oncoming={oncoming_asset}, passer={passer_asset}")
    print("")
    print("Proposed sim.yaml agents block:")
    print(yaml.safe_dump({"agents": agents_block}, sort_keys=False))

    if args.write_config:
        _patch_sim_yaml_agents(Path(args.config), agents_block)
        print(f"Patched {args.config} with agents block")
    else:
        print("Config not modified (use --write-config to patch sim.yaml).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
