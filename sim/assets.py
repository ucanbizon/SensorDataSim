"""Load and validate preprocessed data into SceneAssets."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


@dataclass
class SceneAssets:
    static_points: np.ndarray      # (N, 3) float64, map frame
    static_colors: np.ndarray      # (N, 3) uint8
    static_labels: np.ndarray      # (N,) int32
    static_intensity: np.ndarray   # (N,) float32

    ground_plane: np.ndarray       # [a, b, c, d] float64
    class_map: dict[str, int]      # semantic label name -> integer id
    label_car_id: int | None = None
    static_cars_fullres_points: np.ndarray | None = None  # optional camera-quality subset
    static_cars_fullres_colors: np.ndarray | None = None  # optional camera-quality subset
    dynamic_agent_assets: dict[str, "AgentAsset"] = field(default_factory=dict)
    static_kd_tree: cKDTree | None = None                # XY spatial index for culling
    static_cars_fullres_kd_tree: cKDTree | None = None    # XY spatial index for culling


@dataclass
class AgentAsset:
    name: str
    points: np.ndarray      # (N,3) float64 local frame
    colors: np.ndarray      # (N,3) uint8
    intensity: np.ndarray   # (N,) float32
    meta: dict[str, Any]    # metadata json from asset export


def _load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    # Fail early with a clear message; downstream shape errors are harder to read.
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in {path.parent}")
    pcd = o3d.io.read_point_cloud(str(path))
    if np.asarray(pcd.points).shape[0] == 0:
        raise ValueError(f"Point cloud {path.name} is empty")
    return pcd


def _load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in {path.parent}")
    return np.load(str(path))


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in {path.parent}")
    with open(path) as f:
        return json.load(f)


def _load_optional_static_cars_fullres_npz(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load optional full-res static-car subset for hybrid camera rendering."""
    if not path.exists():
        return None, None
    arr = np.load(str(path))
    if "points" not in arr or "colors" not in arr:
        raise ValueError(f"{path.name} missing required arrays 'points' and 'colors'")
    # Keep dtypes explicit so renderers do not pay conversion costs later.
    pts = np.asarray(arr["points"], dtype=np.float64)
    cols = np.asarray(arr["colors"], dtype=np.uint8)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{path.name} points shape invalid: {pts.shape}")
    if cols.shape != pts.shape:
        raise ValueError(f"{path.name} colors shape {cols.shape} != points shape {pts.shape}")
    return pts, cols


def _load_dynamic_agent_assets(data_dir: Path, asset_map: dict[str, str] | None) -> dict[str, AgentAsset]:
    """Load selected dynamic agent assets from data_dir/agent_assets.

    Args:
      asset_map: mapping of runtime agent frame/key -> asset_name stem
    """
    if not asset_map:
        return {}

    root = data_dir / "agent_assets"
    if not root.exists():
        raise FileNotFoundError(
            f"agent_assets directory not found: {root}. "
            f"Run scripts/build_car_asset_library.py first."
        )

    out: dict[str, AgentAsset] = {}
    for agent_key, asset_name in asset_map.items():
        ply_path = root / f"{asset_name}.ply"
        inten_path = root / f"{asset_name}_intensity.npy"
        meta_path = root / f"{asset_name}_meta.json"

        pcd = _load_point_cloud(ply_path)
        pts = np.asarray(pcd.points)
        cols_f = np.asarray(pcd.colors)
        # open3d stores colors in [0,1] float; camera renderer expects uint8 RGB.
        cols = (cols_f * 255).clip(0, 255).astype(np.uint8)
        inten = _load_npy(inten_path).astype(np.float32, copy=False)
        meta = _load_metadata(meta_path)

        if pts.shape[0] != inten.shape[0]:
            raise ValueError(
                f"Agent asset '{asset_name}' mismatch: {pts.shape[0]} pts vs {inten.shape[0]} intensity"
            )
        out[agent_key] = AgentAsset(
            name=asset_name,
            points=pts,
            colors=cols,
            intensity=inten,
            meta=meta,
        )
    return out


def load_assets(
    data_dir: str | Path,
    fullres: bool = False,
    dynamic_agent_asset_map: dict[str, str] | None = None,
    build_spatial_index: bool = True,
) -> SceneAssets:
    """Load all preprocessed files and validate alignment.

    Args:
        fullres: If True, load scene_static_fullres.ply (skip metadata count check).
    """
    data_dir = Path(data_dir)

    # Phase 1: metadata (class ids, fitted ground plane, preprocessing counts).
    meta = _load_metadata(data_dir / "metadata.json")

    # Phase 2: core static scene arrays (map-frame geometry used by both sensors).
    suffix = "_fullres" if fullres else ""
    scene_pcd = _load_point_cloud(data_dir / f"scene_static{suffix}.ply")
    static_points = np.asarray(scene_pcd.points)
    # Store colors as uint8 once; avoids repeated float->uint8 conversions later.
    static_colors = (np.asarray(scene_pcd.colors) * 255).clip(0, 255).astype(np.uint8)
    static_labels = _load_npy(data_dir / f"scene_labels{suffix}.npy")
    static_intensity = _load_npy(data_dir / f"scene_intensity{suffix}.npy")

    gp = meta["ground_plane"]
    ground_plane = np.array([gp["a"], gp["b"], gp["c"], gp["d"]], dtype=np.float64)
    class_map = dict(meta["class_map"])

    # Phase 3: optional high-quality camera subset + optional dynamic agent assets.
    static_cars_fullres_points, static_cars_fullres_colors = _load_optional_static_cars_fullres_npz(
        data_dir / "static_cars_fullres.npz"
    )
    dynamic_agent_assets = _load_dynamic_agent_assets(data_dir, dynamic_agent_asset_map)

    # Phase 4: assemble one typed container used throughout the runtime.
    assets = SceneAssets(
        static_points=static_points,
        static_colors=static_colors,
        static_labels=static_labels,
        static_intensity=static_intensity,
        ground_plane=ground_plane,
        class_map=class_map,
        label_car_id=int(class_map["Car"]) if "Car" in class_map else None,
        static_cars_fullres_points=static_cars_fullres_points,
        static_cars_fullres_colors=static_cars_fullres_colors,
        dynamic_agent_assets=dynamic_agent_assets,
    )

    # Phase 5: shape/range checks and optional KD-tree acceleration.
    _validate_assets(assets, meta, check_metadata_count=not fullres)

    if build_spatial_index:
        _build_spatial_indices(assets)

    return assets


def _build_spatial_indices(assets: SceneAssets) -> None:
    """Build XY KD-trees for spatial culling.

    The renderers only need a local spatial subset near ego, so KD-trees are
    built on XY (not XYZ) to match the cull rule and reduce query cost.
    """
    t0 = time.perf_counter()
    assets.static_kd_tree = cKDTree(assets.static_points[:, :2])
    dt = time.perf_counter() - t0
    print(f"  KD-tree (static): {assets.static_points.shape[0]:,} pts in {dt:.1f}s")

    if assets.static_cars_fullres_points is not None:
        t0 = time.perf_counter()
        assets.static_cars_fullres_kd_tree = cKDTree(assets.static_cars_fullres_points[:, :2])
        dt = time.perf_counter() - t0
        print(f"  KD-tree (fullres cars): {assets.static_cars_fullres_points.shape[0]:,} pts in {dt:.1f}s")


def _validate_assets(
    assets: SceneAssets,
    meta: dict[str, Any],
    check_metadata_count: bool = True,
) -> None:
    """Run asset validation checks."""
    N = assets.static_points.shape[0]

    assert assets.static_labels.shape == (N,)
    assert assets.static_intensity.shape == (N,)
    assert assets.ground_plane[2] > 0, "Ground normal not upward"
    assert not np.any(np.isnan(assets.static_points))

    if check_metadata_count:
        expected = meta["static_scene"]["point_counts"]["after_voxel_downsample"]
        assert N == expected, f"Scene points {N} != metadata {expected}"

    extra = ""
    if assets.static_cars_fullres_points is not None:
        extra = f", {assets.static_cars_fullres_points.shape[0]:,} fullres car pts"
    dyn = ""
    if assets.dynamic_agent_assets:
        dyn_pts = sum(a.points.shape[0] for a in assets.dynamic_agent_assets.values())
        dyn = f", {len(assets.dynamic_agent_assets)} dyn assets ({dyn_pts:,} pts)"
    print(f"  Assets validated: {N:,} static pts{extra}{dyn}")
