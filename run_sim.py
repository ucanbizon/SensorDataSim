"""Full 15-second simulation -> MCAP export.

Renders VLP-16 LiDAR (10 Hz), pinhole camera (10 Hz), and TF (50 Hz)
for the configured duration, streaming into a single MCAP file.

Usage:
    conda run -n sensorsim python run_sim.py               # downsampled
    conda run -n sensorsim python run_sim.py --fullres     # full-resolution
    conda run -n sensorsim python run_sim.py --output data/processed/sim_output_check.mcap
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sim.assets import load_assets
from sim.camera import render_camera_frame
from sim.config import load_config
from sim.lidar import render_lidar_frame
from sim.mcap_writer import SimWriter
from sim.road_surface import build_road_surface_model_from_assets
from sim.scene_compose import SpatialCullCache, compose_scene
from sim.tf_tree import TFTree
from sim.timeline import EventType, build_timeline
from sim.trajectory import build_trajectory

CONFIG_PATH = "sim.yaml"


@dataclass
class TimelineCounts:
    """Counts of how many timeline ticks contain each event type."""

    tf: int
    lidar: int
    camera: int


@dataclass
class SimulationStats:
    """Run stats collected during the event loop."""

    lidar_times: list[float] = field(default_factory=list)
    camera_times: list[float] = field(default_factory=list)
    lidar_count: int = 0
    camera_count: int = 0
    cull_logged: bool = False

    def record_lidar(self, dt_s: float) -> None:
        self.lidar_times.append(float(dt_s))
        self.lidar_count += 1

    def record_camera(self, dt_s: float) -> None:
        self.camera_times.append(float(dt_s))
        self.camera_count += 1


def _build_dynamic_agent_runtimes(cfg, assets, road_height_fn=None):
    """Build trajectory runtime state for enabled dynamic agents.

    Returns dict keyed by agent frame_id:
      {
        frame_id: {
          "pose_fn": callable,
          "max_t": float,
          "start_delay_s": float,
          "asset_name": str,
        }
      }
    """
    runtimes = {}
    for name, agent_cfg in getattr(cfg, "agents", {}).items():
        if not agent_cfg.enabled:
            continue
        frame_id = agent_cfg.frame_id
        pose_fn, max_t = build_trajectory(
            agent_cfg.waypoints,
            agent_cfg.speed_mps,
            assets.ground_plane,
            z_fn=road_height_fn,
        )
        runtimes[frame_id] = {
            "name": name,
            "asset_name": agent_cfg.asset_name,
            "pose_fn": pose_fn,
            "max_t": max_t,
            "start_delay_s": float(agent_cfg.start_delay_s),
        }
    return runtimes


def _active_agent_transforms(t: float, agent_runtimes: dict[str, dict]) -> dict[str, np.ndarray]:
    """Return T_map_agent transforms for agents that are active at time t."""
    out: dict[str, np.ndarray] = {}
    for frame_id, rt in agent_runtimes.items():
        tau = t - rt["start_delay_s"]
        # Clamp each agent to its own trajectory duration so TF publication stops
        # after the configured path finishes.
        if tau < 0.0 or tau > rt["max_t"]:
            continue
        out[frame_id] = rt["pose_fn"](tau)
    return out


def _count_timeline_events(timeline) -> TimelineCounts:
    """Count how many timeline ticks contain TF / LiDAR / camera events."""
    return TimelineCounts(
        tf=sum(1 for e in timeline if EventType.TF in e.events),
        lidar=sum(1 for e in timeline if EventType.LIDAR in e.events),
        camera=sum(1 for e in timeline if EventType.CAMERA in e.events),
    )


def _resolve_output_path(cfg, args) -> Path:
    """Resolve the output path (default under cfg.data_dir unless overridden)."""
    if args.output is not None:
        output_path = Path(args.output)
    else:
        suffix = "_fullres" if args.fullres else ""
        output_path = Path(cfg.data_dir) / f"sim_output{suffix}.mcap"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _print_header(fullres: bool) -> None:
    """Print the top banner showing the export mode."""
    mode = "FULL-RES" if fullres else "DOWNSAMPLED"
    print("=" * 70)
    print(f"SENSOR DATA SIMULATOR - {mode} MCAP EXPORT")
    print("=" * 70)


def _print_setup_summary(
    cfg,
    *,
    road_surface,
    timeline,
    timeline_counts: TimelineCounts,
    output_path: Path,
    agent_runtimes: dict[str, dict],
    cull_radius_m: float,
) -> None:
    """Print the key run settings after config/assets are loaded."""
    tl = cfg.timeline
    print(
        "  Road surface model: "
        f"{road_surface.n_points:,} pts ({', '.join(road_surface.labels_used)}), "
        f"KD-tree {road_surface.build_time_s:.1f}s, "
        f"r={road_surface.radius_m:.1f}m, k={road_surface.k_neighbors}"
    )
    print(f"  Duration: {tl.duration_s}s")
    print(
        f"  Timeline: {len(timeline)} events "
        f"(TF={timeline_counts.tf}, LiDAR={timeline_counts.lidar}, Camera={timeline_counts.camera})"
    )
    # This runner currently writes compressed images and publishes them on
    # /camera/image_raw/compressed. The config string is the compression format.
    print(f"  Image format: {cfg.camera_render.image_format} (compressed)")
    print(f"  Output: {output_path}")
    if agent_runtimes:
        print(
            f"  Dynamic agents: {len(agent_runtimes)} "
            f"({', '.join(sorted(agent_runtimes.keys()))})"
        )
    print(f"  Cull radius: {cull_radius_m:.0f}m")


def _write_tf_static_once(w: SimWriter, tf_tree: TFTree, ego_fn, agent_runtimes: dict[str, dict]) -> None:
    """Publish the static TF tree once, anchored to the initial ego pose."""
    first_agents = _active_agent_transforms(0.0, agent_runtimes)
    first_snap = tf_tree.snapshot(0.0, ego_fn(0.0), T_map_agents=first_agents)
    w.write_tf_static(0, first_snap)


def _log_initial_culling_once(scene, assets, cull_radius_m: float, stats: SimulationStats) -> None:
    """Print a one-time summary of the static point culling result."""
    if scene is None or stats.cull_logged:
        return
    n_full = assets.static_points.shape[0]
    n_culled = scene.static_points.shape[0]
    pct = 100.0 * n_culled / n_full if n_full > 0 else 0.0
    print(
        f"  Culling: {n_full:,} -> {n_culled:,} static pts "
        f"({pct:.1f}%), radius={cull_radius_m:.0f}m"
    )
    stats.cull_logged = True


def _render_and_write_timeline(
    timeline,
    *,
    assets,
    cfg,
    ego_fn,
    tf_tree: TFTree,
    agent_runtimes: dict[str, dict],
    cull_radius_m: float,
    cull_cache: SpatialCullCache,
    writer: SimWriter,
    wall_start_s: float,
) -> SimulationStats:
    """Render and write all timeline events.

    The timeline is event-driven. A single tick can request TF, LiDAR, and/or
    camera. We compose the scene at most once per tick and reuse it for all
    sensors that fire at that timestamp.
    """
    stats = SimulationStats()
    img_fmt = cfg.camera_render.image_format

    for i, event in enumerate(timeline):
        t = event.t_s
        ns = event.tick_ns

        T_map_base = ego_fn(t)
        T_map_agents = _active_agent_transforms(t, agent_runtimes)
        snap = tf_tree.snapshot(t, T_map_base, T_map_agents=T_map_agents)

        if EventType.TF in event.events:
            writer.write_tf(ns, snap)

        need_scene = (EventType.LIDAR in event.events) or (EventType.CAMERA in event.events)
        scene = (
            compose_scene(
                t,
                assets,
                snap,
                include_agent=bool(agent_runtimes),
                cull_radius_m=cull_radius_m,
                cull_cache=cull_cache,
            )
            if need_scene
            else None
        )
        _log_initial_culling_once(scene, assets, cull_radius_m, stats)

        if EventType.LIDAR in event.events:
            t0 = time.perf_counter()
            frame, _lidar_stats = render_lidar_frame(scene, cfg.lidar_render)
            stats.record_lidar(time.perf_counter() - t0)
            writer.write_pointcloud2(ns, frame)

        if EventType.CAMERA in event.events:
            t0 = time.perf_counter()
            cam_frame, _cam_stats = render_camera_frame(scene, cfg.camera_intrinsics, cfg.camera_render)
            stats.record_camera(time.perf_counter() - t0)
            writer.write_compressed_image(ns, cam_frame, fmt=img_fmt)
            writer.write_camera_info(ns)

        if (i + 1) % 50 == 0 or i == len(timeline) - 1:
            elapsed = time.perf_counter() - wall_start_s
            print(
                f"  [{i+1:4d}/{len(timeline)}] "
                f"sim_t={t:5.2f}s  "
                f"wall={elapsed:.0f}s  "
                f"lidar={stats.lidar_count}  camera={stats.camera_count}"
            )

    return stats


def _print_summary(
    output_path: Path,
    total_time_s: float,
    stats: SimulationStats,
    timeline_counts: TimelineCounts,
) -> None:
    """Print post-run timing and message count summary."""
    file_size = output_path.stat().st_size
    lidar_times = np.asarray(stats.lidar_times, dtype=np.float64)
    camera_times = np.asarray(stats.camera_times, dtype=np.float64)

    print(f"\n-- Summary --")
    print(f"  Total wall time: {total_time_s:.0f}s ({total_time_s/60:.1f} min)")
    print(f"  File: {output_path} ({file_size:,} bytes, {file_size/1024/1024:.1f} MB)")

    print(f"\n  LiDAR: {stats.lidar_count} frames")
    if lidar_times.size > 0:
        print(
            f"    render: mean={lidar_times.mean():.2f}s  "
            f"min={lidar_times.min():.2f}s  max={lidar_times.max():.2f}s"
        )

    print(f"  Camera: {stats.camera_count} frames")
    if camera_times.size > 0:
        print(
            f"    render: mean={camera_times.mean():.2f}s  "
            f"min={camera_times.min():.2f}s  max={camera_times.max():.2f}s"
        )

    print(f"\n  Messages written:")
    print(f"    /tf_static:                    1")
    print(f"    /tf:                           {timeline_counts.tf}")
    print(f"    /velodyne_points:              {stats.lidar_count}")
    print(f"    /camera/image_raw/compressed:  {stats.camera_count}")
    print(f"    /camera/camera_info:{stats.camera_count}")

    print(f"\n  Done. Open in Foxglove Studio:")
    print(f"    foxglove-studio {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sensor data simulator")
    parser.add_argument(
        "--fullres",
        action="store_true",
        help="Use full-resolution (non-downsampled) scene data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output MCAP path (default: data_dir/sim_output[_fullres].mcap)",
    )
    args = parser.parse_args()

    _print_header(args.fullres)

    print("\n-- Loading config and assets --")
    cfg = load_config(CONFIG_PATH)

    dynamic_agent_asset_map = {
        agent_cfg.frame_id: agent_cfg.asset_name
        for agent_cfg in getattr(cfg, "agents", {}).values()
        if getattr(agent_cfg, "enabled", False)
    }
    assets = load_assets(
        cfg.data_dir,
        fullres=args.fullres,
        dynamic_agent_asset_map=dynamic_agent_asset_map,
    )

    # Warm up JIT kernels once so the first rendered frame is not dominated by
    # compilation latency.
    from sim.numba_kernels import warmup_numba

    warmup_numba()

    road_surface = build_road_surface_model_from_assets(assets)
    ego_fn, _ = build_trajectory(
        cfg.ego.waypoints,
        cfg.ego.speed_mps,
        assets.ground_plane,
        z_fn=road_surface.height,
    )
    agent_runtimes = _build_dynamic_agent_runtimes(cfg, assets, road_height_fn=road_surface.height)
    tf_tree = TFTree(cfg)

    tl = cfg.timeline
    timeline = build_timeline(tl.duration_s, tl.lidar_rate_hz, tl.camera_rate_hz, tl.tf_rate_hz)
    timeline_counts = _count_timeline_events(timeline)
    output_path = _resolve_output_path(cfg, args)

    # Use one conservative cull radius for both sensors so compose_scene can be
    # called once per timeline tick and reused by LiDAR and camera.
    cull_radius_m = max(cfg.lidar_render.max_range_m, cfg.camera_render.max_range_m) + 10.0
    cull_cache = SpatialCullCache()

    _print_setup_summary(
        cfg,
        road_surface=road_surface,
        timeline=timeline,
        timeline_counts=timeline_counts,
        output_path=output_path,
        agent_runtimes=agent_runtimes,
        cull_radius_m=cull_radius_m,
    )

    print("\n-- Rendering and writing MCAP --")
    wall_start_s = time.perf_counter()

    with SimWriter(str(output_path), cfg.frame_names, cfg.camera_intrinsics) as writer:
        _write_tf_static_once(writer, tf_tree, ego_fn, agent_runtimes)
        run_stats = _render_and_write_timeline(
            timeline,
            assets=assets,
            cfg=cfg,
            ego_fn=ego_fn,
            tf_tree=tf_tree,
            agent_runtimes=agent_runtimes,
            cull_radius_m=cull_radius_m,
            cull_cache=cull_cache,
            writer=writer,
            wall_start_s=wall_start_s,
        )

    total_time_s = time.perf_counter() - wall_start_s
    _print_summary(output_path, total_time_s, run_stats, timeline_counts)


if __name__ == "__main__":
    main()
