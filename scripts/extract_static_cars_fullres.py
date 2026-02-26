"""Extract full-resolution static car points for hybrid camera rendering.

Creates `data/processed/static_cars_fullres.npz` containing:
  - points: (N, 3) float64 map-frame points
  - colors: (N, 3) uint8 RGB

This lets the camera renderer use high-resolution points only for parked/static
cars while keeping the rest of the scene downsampled.

Usage:
    conda run -n sensorsim python scripts/extract_static_cars_fullres.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.assets import load_assets
from sim.config import load_config


def main() -> int:
    """Extract and store full-resolution static car points for hybrid rendering."""
    # Load the full-resolution static scene once, then keep only `Car`-labeled
    # points. This file is used by the camera renderer for a hybrid quality mode.
    cfg = load_config("sim.yaml")
    assets = load_assets(cfg.data_dir, fullres=True)

    mask = assets.static_labels == assets.label_car_id
    n_car = int(np.count_nonzero(mask))

    pts = np.asarray(assets.static_points[mask], dtype=np.float64)
    cols = np.asarray(assets.static_colors[mask], dtype=np.uint8)

    # Save a compact npz so run-time loading is simple and fast.
    out_path = Path(cfg.data_dir) / "static_cars_fullres.npz"
    np.savez_compressed(str(out_path), points=pts, colors=cols)

    print("=" * 70)
    print("EXTRACT STATIC CARS (FULL-RES)")
    print("=" * 70)
    print(f"Car label ID: {assets.label_car_id}")
    print(f"Car points:   {n_car:,}")
    print(f"Output:       {out_path}")
    print(f"Size:         {out_path.stat().st_size:,} bytes")
    bb_min = pts.min(axis=0)
    bb_max = pts.max(axis=0)
    print(f"BBox min/max: {bb_min.tolist()} / {bb_max.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
