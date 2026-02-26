"""Quick acceptance checks for the simulation pipeline.

Usage:  python -m sim.validate --config sim.yaml
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .assets import load_assets
from .config import load_config
from .tf_tree import TFTree
from .timeline import EventType, build_timeline
from .trajectory import build_trajectory, ground_z
from .transforms import is_valid_se3


def run_validation(config_path: str) -> bool:
    """Run acceptance gates. Returns True if all pass."""
    cfg = load_config(config_path)
    assets = load_assets(cfg.data_dir)
    fails = 0

    def check(ok: bool, name: str, detail: str = ""):
        nonlocal fails
        tag = "PASS" if ok else "FAIL"
        if not ok:
            fails += 1
        # Flat text output keeps this easy to scan in terminal logs.
        print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))

    # -- Timeline --
    print("\n-- Timeline --")
    tl = cfg.timeline
    events = build_timeline(tl.duration_s, tl.lidar_rate_hz,
                            tl.camera_rate_hz, tl.tf_rate_hz)
    n_lidar = sum(1 for e in events if EventType.LIDAR in e.events)
    n_camera = sum(1 for e in events if EventType.CAMERA in e.events)
    n_tf = sum(1 for e in events if EventType.TF in e.events)
    check(n_lidar == int(tl.duration_s * tl.lidar_rate_hz), "LiDAR count", str(n_lidar))
    check(n_camera == int(tl.duration_s * tl.camera_rate_hz), "Camera count", str(n_camera))
    check(n_tf == int(tl.duration_s * tl.tf_rate_hz), "TF count", str(n_tf))

    # -- Ego trajectory --
    print("\n-- Ego trajectory --")
    ego_fn, ego_max_t = build_trajectory(
        cfg.ego.waypoints, cfg.ego.speed_mps, assets.ground_plane)
    check(ego_max_t >= tl.duration_s, "Duration",
          f"{ego_max_t:.1f}s >= {tl.duration_s}s")

    # Sample a few times rather than every frame. This keeps validation quick
    # while still catching broken trajectory or transform math.
    se3_ok = all(is_valid_se3(ego_fn(t), atol=1e-6) for t in [0, 5, 10, 14.9])
    check(se3_ok, "SE3 valid")

    # Loose tolerance: this validator checks the global-plane trajectory path.
    z_ok = all(
        abs(ego_fn(t)[2, 3] - ground_z(ego_fn(t)[0, 3], ego_fn(t)[1, 3], assets.ground_plane)) < 0.5
        for t in [0, 5, 10, 14.9])
    check(z_ok, "Ego on ground")

    # -- TF tree --
    print("\n-- TF tree --")
    tf_tree = TFTree(cfg)
    for t in [0.0, 7.5, 14.9]:
        snap = tf_tree.snapshot(t, ego_fn(t))
        vel_h = snap.T_map_velodyne[2, 3] - ground_z(
            snap.T_map_velodyne[0, 3], snap.T_map_velodyne[1, 3], assets.ground_plane)
        check(1.3 <= vel_h <= 2.3, f"Velodyne height t={t}", f"{vel_h:.2f}m")

        cam_fwd = snap.T_map_camera_optical[:3, 2]
        ego_fwd = ego_fn(t)[:3, 0]
        dot = float(np.dot(cam_fwd, ego_fwd))
        check(dot > 0.9, f"Camera forward t={t}", f"dot={dot:.3f}")

    # -- Summary --
    print(f"\n  {'PASS' if fails == 0 else 'FAIL'}: {fails} failures")
    return fails == 0


def main():
    p = argparse.ArgumentParser(description="Validate simulation core")
    p.add_argument("--config", default="sim.yaml")
    args = p.parse_args()
    sys.exit(0 if run_validation(args.config) else 1)
