"""Build a library of high-quality dynamic car assets from Toronto-3D raw tiles.

This script scans one or more raw Toronto-3D tiles (e.g. L001/L002/L003),
extracts `Car`-labeled point clusters, scores cluster quality (to reject
roof-only/merged/truncated clusters), and exports the best candidates as local-
frame car assets suitable for use as dynamic agents in the simulator.

Outputs (under data/processed/agent_assets by default):
  - *.ply              (local-frame car points with RGB)
  - *_intensity.npy    (aligned intensity values)
  - *_meta.json        (cluster metadata + quality metrics + local transform)
  - asset_catalog.json (all analyzed clusters + exported subset)

Local frame convention for exported car assets:
  - X forward (PCA longitudinal axis, deterministic sign)
  - Y left
  - Z up (global +Z)
  - Origin = bottom-center (centroid XY, z at 1st percentile)

Usage:
  conda run -n sensorsim python scripts/build_car_asset_library.py
  conda run -n sensorsim python scripts/build_car_asset_library.py --tiles L001,L003 --export-top 6
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from plyfile import PlyData


# Reuse the same broad plausibility envelope as preprocessing.
CAR_LENGTH_RANGE = (3.0, 6.5)
CAR_WIDTH_RANGE = (1.2, 2.8)
CAR_HEIGHT_RANGE = (1.0, 2.5)

DBSCAN_EPS = 0.30
DBSCAN_MIN_POINTS = 100
DBSCAN_FALLBACK_MIN_POINTS = (50, 20)
CLUSTER_VOXEL_SIZE_M = 0.15


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """One candidate car cluster transformed into a local asset frame."""

    tile: str
    cluster_id: int
    n_points: int
    local_xyz: np.ndarray
    rgb: np.ndarray
    intensity: np.ndarray
    origin_map: np.ndarray
    R_map_to_local: np.ndarray
    metrics: dict[str, Any]
    score: float
    plausible: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build dynamic car asset library from raw Toronto-3D tiles.")
    p.add_argument("--tiles", default="L001,L002,L003", help="Comma-separated tile names (without .ply)")
    p.add_argument("--raw-dir", default="data/raw/Toronto_3D", help="Directory containing raw Toronto-3D .ply tiles")
    p.add_argument("--class-map", default="data/raw/Toronto_3D/Mavericks_classes_9.txt", help="Class map file")
    p.add_argument("--out-dir", default="data/processed/agent_assets", help="Output directory for exported assets")
    p.add_argument("--export-top", type=int, default=8, help="Export top N clusters across all tiles")
    p.add_argument("--top-per-tile", type=int, default=4, help="Max exported assets per tile before global cap")
    p.add_argument("--dbscan-eps", type=float, default=DBSCAN_EPS)
    p.add_argument("--dbscan-min-points", type=int, default=DBSCAN_MIN_POINTS)
    p.add_argument(
        "--cluster-voxel-m",
        type=float,
        default=CLUSTER_VOXEL_SIZE_M,
        help="Voxel size used for clustering car points (0 disables downsample+trace clustering)",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing asset files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Raw input parsing and geometry helpers
# ---------------------------------------------------------------------------

def parse_class_map(path: Path) -> dict[str, int]:
    # Parse "ClassName <int>" lines into a simple lookup table.
    out: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                continue
            name, value = parts[0], int(parts[1])
            out[name] = value
    return out


def _safe_hist_entropy(frac: np.ndarray) -> float:
    """Normalized entropy in [0, 1] for a histogram of occupancy fractions."""
    frac = frac[frac > 0]
    if frac.size == 0:
        return 0.0
    entropy = -np.sum(frac * np.log2(frac))
    return float(entropy / np.log2(max(2, frac.size)))


def _deterministic_pca_xy(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Return PC1 in XY (unit vector) and explained variance ratio."""
    centered = xy - xy.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, -1]
    # PCA eigenvectors have arbitrary sign. Fix the sign so exported local
    # frames are stable across runs (reproducible metadata and filenames).
    if pc1[0] < 0:
        pc1 = -pc1
    var_ratio = float(eigvals[-1] / max(np.sum(eigvals), 1e-12))
    return pc1.astype(np.float64), var_ratio


def make_local_frame(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Build local car frame and transform points into it.

    Returns:
      local_xyz, origin_map(3,), R_map_to_local(3,3), frame_metrics
    """
    xy = points_xyz[:, :2]
    pc1_xy, pca_var = _deterministic_pca_xy(xy)

    X_local = np.array([pc1_xy[0], pc1_xy[1], 0.0], dtype=np.float64)
    X_local /= np.linalg.norm(X_local)
    Z_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    Y_local = np.cross(Z_local, X_local)
    Y_local /= np.linalg.norm(Y_local)

    # Rows are local axes expressed in map coordinates, so this matrix maps
    # map-frame vectors into the local car frame by left-multiplication.
    R_map_to_local = np.stack([X_local, Y_local, Z_local], axis=0)

    centroid_xy = xy.mean(axis=0)
    # Use a percentile instead of min() so a few outlier points do not pull the
    # local origin below the car body.
    z_origin = float(np.percentile(points_xyz[:, 2], 1))  # robust bottom estimate
    origin_map = np.array([centroid_xy[0], centroid_xy[1], z_origin], dtype=np.float64)

    local_xyz = (R_map_to_local @ (points_xyz - origin_map).T).T

    yaw_deg = float(np.degrees(np.arctan2(X_local[1], X_local[0])))
    frame_metrics = {
        "pca_variance_explained": pca_var,
        "yaw_deg_source_frame": yaw_deg,
        "z_origin_percentile1": z_origin,
    }
    return local_xyz, origin_map, R_map_to_local, frame_metrics


def compute_quality_metrics(
    local_xyz: np.ndarray,
    tile_bbox_xy: tuple[np.ndarray, np.ndarray],
    cluster_bbox_map: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Compute geometric/completeness metrics and a quality score.

    The metrics are intentionally heuristic. The goal is not perfect car
    recognition, but to rank clusters by how useful they are as simulator
    assets (complete shape, plausible dimensions, not truncated).
    """
    mins = local_xyz.min(axis=0)
    maxs = local_xyz.max(axis=0)
    dims = maxs - mins
    dx, dy, dz = [float(v) for v in dims]
    length = max(dx, dy)
    width = min(dx, dy)
    height = dz

    plausible_dims = (
        CAR_LENGTH_RANGE[0] <= length <= CAR_LENGTH_RANGE[1]
        and CAR_WIDTH_RANGE[0] <= width <= CAR_WIDTH_RANGE[1]
        and CAR_HEIGHT_RANGE[0] <= height <= CAR_HEIGHT_RANGE[1]
    )

    # Normalize into [0,1] box for occupancy/distribution metrics.
    eps = 1e-9
    xn = (local_xyz[:, 0] - mins[0]) / max(dx, eps)
    yn = (local_xyz[:, 1] - mins[1]) / max(dy, eps)
    zn = (local_xyz[:, 2] - mins[2]) / max(dz, eps)

    xcn = 2.0 * (xn - 0.5)
    ycn = 2.0 * (yn - 0.5)

    roof_frac = float(np.mean(zn > 0.80))
    lower_frac = float(np.mean(zn < 0.35))
    side_frac = float(np.mean((np.abs(ycn) > 0.60) & (zn < 0.90)))
    end_frac = float(np.mean((np.abs(xcn) > 0.60) & (zn < 0.90)))

    z_hist = np.histogram(zn, bins=8, range=(0.0, 1.0))[0].astype(np.float64)
    z_hist_frac = z_hist / max(z_hist.sum(), 1.0)
    vertical_entropy = _safe_hist_entropy(z_hist_frac)

    # Coarse voxel occupancy approximates "surface richness" and helps reject
    # sparse or heavily truncated clusters.
    gx, gy, gz = 12, 8, 6
    ix = np.clip((xn * gx).astype(np.int32), 0, gx - 1)
    iy = np.clip((yn * gy).astype(np.int32), 0, gy - 1)
    iz = np.clip((zn * gz).astype(np.int32), 0, gz - 1)
    vox = np.stack([ix, iy, iz], axis=1)
    vox_unique = np.unique(vox, axis=0)
    occ_frac = float(vox_unique.shape[0] / (gx * gy * gz))

    # "Roof-only" heuristic: if roof dominates and side/lower support is weak.
    roof_only_score = float(max(0.0, roof_frac - 0.65) * 2.0 + max(0.0, 0.15 - side_frac) * 3.0)
    roof_only_suspect = bool(roof_frac > 0.70 and side_frac < 0.12 and lower_frac < 0.20)

    # Tile edge truncation heuristic (XY only).
    tile_min_xy, tile_max_xy = tile_bbox_xy
    cmin_map, cmax_map = cluster_bbox_map
    edge_margin_xy = np.min(
        np.array(
            [
                cmin_map[0] - tile_min_xy[0],
                cmin_map[1] - tile_min_xy[1],
                tile_max_xy[0] - cmax_map[0],
                tile_max_xy[1] - cmax_map[1],
            ],
            dtype=np.float64,
        )
    )
    edge_truncation_suspect = bool(edge_margin_xy < 1.0)

    # Quality score (higher is better). Weights favor complete side/lower geometry.
    score = 0.0
    score += 2.0 * float(plausible_dims)
    score += 1.5 * side_frac
    score += 1.0 * end_frac
    score += 0.8 * lower_frac
    score += 0.8 * vertical_entropy
    score += 0.7 * occ_frac
    score -= 1.0 * roof_frac
    score -= 2.0 * roof_only_score
    score -= 1.2 * float(edge_truncation_suspect)

    return {
        "dimensions_local": {"dx": dx, "dy": dy, "dz": dz},
        "dimensions_sorted": {"length": float(length), "width": float(width), "height": float(height)},
        "plausible_dims": plausible_dims,
        "roof_frac": roof_frac,
        "lower_frac": lower_frac,
        "side_frac": side_frac,
        "end_frac": end_frac,
        "vertical_entropy": vertical_entropy,
        "occupancy_frac": occ_frac,
        "roof_only_suspect": roof_only_suspect,
        "edge_margin_xy_m": float(edge_margin_xy),
        "edge_truncation_suspect": edge_truncation_suspect,
        "quality_score": float(score),
    }


def save_rgb_ply(points_local: np.ndarray, rgb: np.ndarray, out_path: Path) -> None:
    # Open3D already raises/prints useful errors; keep this writer minimal.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points_local, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(rgb, dtype=np.float64) / 255.0)
    o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False)


def export_cluster_asset(cluster: ClusterResult, out_dir: Path, overwrite: bool) -> dict[str, Any]:
    """Export local-frame asset files and metadata for one cluster."""
    stem = f"{cluster.tile.lower()}_car_cluster{cluster.cluster_id:03d}_q{cluster.score:.3f}"
    ply_path = out_dir / f"{stem}.ply"
    inten_path = out_dir / f"{stem}_intensity.npy"
    meta_path = out_dir / f"{stem}_meta.json"

    if not overwrite and (ply_path.exists() or inten_path.exists() or meta_path.exists()):
        raise FileExistsError(f"Asset files already exist for {stem}. Use --overwrite.")

    save_rgb_ply(cluster.local_xyz, cluster.rgb, ply_path)
    np.save(str(inten_path), np.asarray(cluster.intensity, dtype=np.float32))

    T_map_local = np.eye(4, dtype=np.float64)
    T_map_local[:3, :3] = cluster.R_map_to_local.T
    T_map_local[:3, 3] = cluster.origin_map

    meta = {
        "asset_name": stem,
        "source_tile": cluster.tile,
        "source_cluster_id": int(cluster.cluster_id),
        "n_points": int(cluster.n_points),
        "origin_map": cluster.origin_map.tolist(),
        "T_map_asset_local": T_map_local.tolist(),
        "R_map_to_local": cluster.R_map_to_local.tolist(),
        "metrics": cluster.metrics,
        "quality_score": float(cluster.score),
        "plausible": bool(cluster.plausible),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "asset_name": stem,
        "ply": str(ply_path.as_posix()),
        "intensity_npy": str(inten_path.as_posix()),
        "meta_json": str(meta_path.as_posix()),
        "source_tile": cluster.tile,
        "source_cluster_id": int(cluster.cluster_id),
        "quality_score": float(cluster.score),
    }


# ---------------------------------------------------------------------------
# Tile loading and clustering
# ---------------------------------------------------------------------------

def load_tile_raw(tile_path: Path) -> dict[str, np.ndarray]:
    """Load only the fields I need from a raw Toronto-3D PLY."""
    ply = PlyData.read(str(tile_path))
    v = ply["vertex"]

    # Keep names explicit so it is easy to map back to the source PLY schema.
    x = np.asarray(v["x"], dtype=np.float64)
    y = np.asarray(v["y"], dtype=np.float64)
    z = np.asarray(v["z"], dtype=np.float64)
    rgb = np.column_stack(
        [
            np.asarray(v["red"], dtype=np.uint8),
            np.asarray(v["green"], dtype=np.uint8),
            np.asarray(v["blue"], dtype=np.uint8),
        ]
    )
    intensity = np.asarray(v["scalar_Intensity"], dtype=np.float32)
    labels = np.round(np.asarray(v["scalar_Label"], dtype=np.float32)).astype(np.int32)

    xyz = np.column_stack([x, y, z])
    return {"xyz": xyz, "rgb": rgb, "intensity": intensity, "labels": labels}


def cluster_car_points(
    tile_name: str,
    xyz: np.ndarray,
    rgb: np.ndarray,
    intensity: np.ndarray,
    labels: np.ndarray,
    label_car_id: int,
    dbscan_eps: float,
    dbscan_min_points: int,
    cluster_voxel_m: float,
) -> list[ClusterResult]:
    """Cluster `Car`-labeled points and score each cluster as an asset candidate."""
    car_mask = labels == label_car_id
    car_xyz = xyz[car_mask]
    car_rgb = rgb[car_mask]
    car_int = intensity[car_mask]
    if car_xyz.shape[0] == 0:
        return []

    car_pcd = o3d.geometry.PointCloud()
    car_pcd.points = o3d.utility.Vector3dVector(car_xyz)

    use_trace_clustering = cluster_voxel_m > 0.0
    if use_trace_clustering:
        # Downsample only for DBSCAN speed, then map clustered voxels back to
        # original points using the voxel trace index lists.
        min_bound = car_xyz.min(axis=0) - 1e-3
        max_bound = car_xyz.max(axis=0) + 1e-3
        ds_pcd, _, trace = car_pcd.voxel_down_sample_and_trace(
            float(cluster_voxel_m), min_bound, max_bound
        )
        cluster_input_pcd = ds_pcd
        print(
            f"  {tile_name}: car pts {car_xyz.shape[0]:,} -> cluster voxels {len(trace):,} "
            f"(voxel={cluster_voxel_m:.2f}m)"
        )
    else:
        cluster_input_pcd = car_pcd
        trace = None

    cluster_labels = np.array(
        cluster_input_pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points, print_progress=False)
    )
    n_clusters = int(cluster_labels.max() + 1) if cluster_labels.size else 0
    if n_clusters == 0:
        # Fallbacks recover smaller but still useful cars if the default
        # min_points is too strict for a given tile.
        for fallback in DBSCAN_FALLBACK_MIN_POINTS:
            cluster_labels = np.array(
                cluster_input_pcd.cluster_dbscan(eps=dbscan_eps, min_points=fallback, print_progress=False)
            )
            n_clusters = int(cluster_labels.max() + 1) if cluster_labels.size else 0
            if n_clusters > 0:
                break
    if n_clusters == 0:
        return []

    tile_min_xy = xyz[:, :2].min(axis=0)
    tile_max_xy = xyz[:, :2].max(axis=0)

    results: list[ClusterResult] = []
    for cid in range(n_clusters):
        if trace is not None:
            ds_idx = np.flatnonzero(cluster_labels == cid)
            if ds_idx.size == 0:
                continue
            src_chunks = []
            for i_ds in ds_idx.tolist():
                src = np.asarray(trace[i_ds], dtype=np.int64)
                if src.size:
                    src_chunks.append(src[src >= 0])
            if not src_chunks:
                continue
            src_idx = np.unique(np.concatenate(src_chunks))
            pts = car_xyz[src_idx]
            cols = car_rgb[src_idx]
            ints = car_int[src_idx]
        else:
            mask = cluster_labels == cid
            pts = car_xyz[mask]
            cols = car_rgb[mask]
            ints = car_int[mask]
        if pts.shape[0] < 200:
            continue

        local_xyz, origin_map, R_map_to_local, frame_metrics = make_local_frame(pts)
        cmin = pts.min(axis=0)
        cmax = pts.max(axis=0)
        metrics = compute_quality_metrics(local_xyz, (tile_min_xy, tile_max_xy), (cmin, cmax))
        metrics.update(frame_metrics)
        metrics["centroid_map"] = pts.mean(axis=0).tolist()
        metrics["bbox_min_map"] = cmin.tolist()
        metrics["bbox_max_map"] = cmax.tolist()

        plausible = bool(metrics["plausible_dims"]) and not bool(metrics["edge_truncation_suspect"])
        # Slight tie-break toward denser clusters when heuristic scores are equal.
        score = float(metrics["quality_score"] + 0.00002 * pts.shape[0])

        results.append(
            ClusterResult(
                tile=tile_name,
                cluster_id=cid,
                n_points=int(pts.shape[0]),
                local_xyz=local_xyz.astype(np.float64),
                rgb=cols.astype(np.uint8),
                intensity=ints.astype(np.float32),
                origin_map=origin_map,
                R_map_to_local=R_map_to_local,
                metrics=metrics,
                score=score,
                plausible=plausible,
            )
        )
    return results


def summarize_cluster_for_catalog(c: ClusterResult) -> dict[str, Any]:
    """Serialize a cluster result into a compact catalog entry."""
    return {
        "tile": c.tile,
        "cluster_id": int(c.cluster_id),
        "n_points": int(c.n_points),
        "score": float(c.score),
        "plausible": bool(c.plausible),
        "metrics": c.metrics,
    }


def main() -> int:
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    class_map_path = Path(args.class_map)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_map = parse_class_map(class_map_path)
    label_car = class_map["Car"]

    tiles = [t.strip() for t in args.tiles.split(",") if t.strip()]

    print("=" * 78)
    print("BUILD CAR ASSET LIBRARY")
    print("=" * 78)
    print(f"Tiles: {tiles}")
    print(f"Raw dir: {raw_dir}")
    print(f"Output: {out_dir}")
    print(f"Car label ID: {label_car}")
    print(f"DBSCAN: eps={args.dbscan_eps}, min_points={args.dbscan_min_points}")
    print(f"Cluster voxel: {args.cluster_voxel_m:.2f} m" if args.cluster_voxel_m > 0 else "Cluster voxel: disabled")

    all_clusters: list[ClusterResult] = []
    by_tile: dict[str, list[ClusterResult]] = {}

    # ------------------------------------------------------------------
    # Phase 1: scan each tile, cluster car points, and score clusters
    # ------------------------------------------------------------------
    for tile in tiles:
        tile_path = raw_dir / f"{tile}.ply"
        print("\n" + "-" * 78)
        print(f"Tile {tile}: {tile_path}")
        print("-" * 78)
        data = load_tile_raw(tile_path)
        xyz = data["xyz"]
        labels = data["labels"]
        n_total = xyz.shape[0]
        n_car = int(np.count_nonzero(labels == label_car))
        print(f"Loaded {n_total:,} points; car-labeled points: {n_car:,}")

        clusters = cluster_car_points(
            tile_name=tile,
            xyz=xyz,
            rgb=data["rgb"],
            intensity=data["intensity"],
            labels=labels,
            label_car_id=label_car,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_points=args.dbscan_min_points,
            cluster_voxel_m=args.cluster_voxel_m,
        )
        clusters.sort(key=lambda c: c.score, reverse=True)
        by_tile[tile] = clusters
        all_clusters.extend(clusters)

        print(f"Clusters analyzed: {len(clusters)}")
        for c in clusters[: min(8, len(clusters))]:
            dims = c.metrics["dimensions_sorted"]
            print(
                f"  #{c.cluster_id:03d} score={c.score:6.3f} "
                f"pts={c.n_points:6d} plausible={int(c.plausible)} "
                f"dims={dims['length']:.2f}x{dims['width']:.2f}x{dims['height']:.2f} "
                f"roof={c.metrics['roof_frac']:.2f} side={c.metrics['side_frac']:.2f}"
            )

        # Drop large raw arrays before loading the next tile.
        del data, xyz, labels

    # ------------------------------------------------------------------
    # Phase 2: choose which clusters to export as reusable assets
    # ------------------------------------------------------------------
    # Select exported assets: per-tile cap then global cap.
    preselected: list[ClusterResult] = []
    for tile in tiles:
        tile_clusters = by_tile.get(tile, [])
        plausible = [c for c in tile_clusters if c.plausible and not c.metrics["roof_only_suspect"]]
        fallback = plausible if plausible else tile_clusters
        preselected.extend(fallback[: args.top_per_tile])

    # Deduplicate by (tile, cluster_id), sort globally, apply global cap.
    uniq: dict[tuple[str, int], ClusterResult] = {(c.tile, c.cluster_id): c for c in preselected}
    selected = sorted(uniq.values(), key=lambda c: c.score, reverse=True)[: args.export_top]

    exported_assets: list[dict[str, Any]] = []
    for c in selected:
        exported_assets.append(export_cluster_asset(c, out_dir, overwrite=args.overwrite))

    # ------------------------------------------------------------------
    # Phase 3: write machine-readable catalog for downstream scripts
    # ------------------------------------------------------------------
    catalog = {
        "version": 1,
        "inputs": {
            "tiles": tiles,
            "raw_dir": str(raw_dir.as_posix()),
            "class_map": str(class_map_path.as_posix()),
            "class_map_values": class_map,
            "dbscan_eps": float(args.dbscan_eps),
            "dbscan_min_points": int(args.dbscan_min_points),
            "cluster_voxel_m": float(args.cluster_voxel_m),
        },
        "selection": {
            "top_per_tile": int(args.top_per_tile),
            "export_top": int(args.export_top),
            "selected_count": len(selected),
        },
        "selected_assets": exported_assets,
        "clusters_by_tile": {
            tile: [summarize_cluster_for_catalog(c) for c in clusters]
            for tile, clusters in by_tile.items()
        },
    }

    catalog_path = out_dir / "asset_catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    print("\n" + "=" * 78)
    print("EXPORT COMPLETE")
    print("=" * 78)
    print(f"Clusters analyzed total: {len(all_clusters)}")
    print(f"Assets exported:         {len(exported_assets)}")
    print(f"Catalog:                {catalog_path}")
    for item in exported_assets:
        print(
            f"  - {item['asset_name']} "
            f"(tile={item['source_tile']}, cluster={item['source_cluster_id']}, q={item['quality_score']:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
