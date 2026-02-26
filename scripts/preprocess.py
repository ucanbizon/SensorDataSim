#!/usr/bin/env python3
"""
Preprocessing pipeline for Toronto-3D L002 → simulation-ready data.

Transforms the raw L002.ply into clean, provenance-tracked, downsampled arrays
with full metadata and QC reporting for a VLP-16 LiDAR + pinhole camera simulation.

Usage:
    conda run -n sensorsim python scripts/preprocess.py [--skip-outlier-removal]
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw/Toronto_3D")
RAW_PLY = RAW_DIR / "L002.ply"
CLASS_MAP_FILE = RAW_DIR / "Mavericks_classes_9.txt"
OUT_DIR = Path("data/processed")
EXPECTED_FILE_SIZE = 442_203_833
EXPECTED_VERTEX_COUNT = 10_283_800

EXPECTED_SCHEMA = [
    ("x", "double"),
    ("y", "double"),
    ("z", "double"),
    ("red", "uchar"),
    ("green", "uchar"),
    ("blue", "uchar"),
    ("scalar_Intensity", "float"),
    ("scalar_GPSTime", "float"),
    ("scalar_ScanAngleRank", "float"),
    ("scalar_Label", "float"),
]

# Processing parameters
VOXEL_SIZE = 0.05          # meters
DBSCAN_EPS = 0.3           # meters
DBSCAN_MIN_POINTS = 100
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 3.0
RANSAC_DIST_THRESHOLD = 0.05
RANSAC_N = 3
RANSAC_ITERATIONS = 1000
ROI_HALF_EXTENT = 200.0    # meters along road axis

# Car dimension plausibility ranges (meters)
CAR_LENGTH_RANGE = (3.0, 6.0)
CAR_WIDTH_RANGE = (1.2, 2.5)
CAR_HEIGHT_RANGE = (1.0, 2.2)

# QC tracking
qc_gates = []


def qc_pass(name: str, detail: str = ""):
    qc_gates.append({"name": name, "status": "PASS", "detail": detail})
    print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))


def qc_fail(name: str, detail: str = ""):
    qc_gates.append({"name": name, "status": "FAIL", "detail": detail})
    print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def qc_warn(name: str, detail: str = ""):
    qc_gates.append({"name": name, "status": "WARN", "detail": detail})
    print(f"  [WARN] {name}" + (f" — {detail}" if detail else ""))


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Integrity, Schema, and Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
def step1_integrity():
    print("\n" + "=" * 70)
    print("STEP 1: Integrity, Schema, and Reproducibility")
    print("=" * 70)

    # File existence
    assert RAW_PLY.exists(), f"Input file not found: {RAW_PLY}"
    qc_pass("File exists", str(RAW_PLY))

    # File size
    actual_size = RAW_PLY.stat().st_size
    assert actual_size == EXPECTED_FILE_SIZE, (
        f"File size mismatch: expected {EXPECTED_FILE_SIZE}, got {actual_size}"
    )
    qc_pass("File size", f"{actual_size:,} bytes")

    # SHA256 checksum
    print("  Computing SHA256 checksum (this may take a moment)...")
    sha256 = hashlib.sha256()
    with open(RAW_PLY, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    checksum = sha256.hexdigest()
    print(f"  SHA256: {checksum}")
    qc_pass("SHA256 computed", checksum)

    # Parse PLY header (without loading data)
    ply = PlyData.read(str(RAW_PLY))
    vertex_elem = ply["vertex"]

    # Verify endianness
    byte_order = ply.byte_order
    assert byte_order == "<", f"Expected little-endian ('<'), got '{byte_order}'"
    qc_pass("Endianness", "binary_little_endian")

    # Verify vertex count
    n_vertices = vertex_elem.count
    assert n_vertices == EXPECTED_VERTEX_COUNT, (
        f"Vertex count mismatch: expected {EXPECTED_VERTEX_COUNT}, got {n_vertices}"
    )
    qc_pass("Vertex count", f"{n_vertices:,}")

    # Verify schema (property order and types)
    # plyfile val_dtype returns numpy dtype codes: f8=float64, f4=float32, u1=uint8
    PLY_TO_NUMPY_DTYPE = {
        "double": "f8", "float": "f4", "uchar": "u1",
    }
    for i, (exp_name, exp_type) in enumerate(EXPECTED_SCHEMA):
        prop = vertex_elem.properties[i]
        assert prop.name == exp_name, (
            f"Property {i} name mismatch: expected '{exp_name}', got '{prop.name}'"
        )
        expected_dtype = PLY_TO_NUMPY_DTYPE[exp_type]
        actual_dtype = str(prop.val_dtype)
        assert actual_dtype == expected_dtype, (
            f"Property '{exp_name}' type mismatch: expected '{expected_dtype}', "
            f"got '{actual_dtype}'"
        )
    n_props = len(vertex_elem.properties)
    qc_pass("Schema", f"{n_props} properties in expected order and types")

    print(f"  Schema verified: binary_little_endian, {n_vertices:,} vertices, "
          f"{n_props} properties")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return ply, checksum


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Parse Class Map and Bind Label Constants
# ──────────────────────────────────────────────────────────────────────────────
def step2_parse_class_map():
    print("\n" + "=" * 70)
    print("STEP 2: Parse Class Map and Bind Label Constants")
    print("=" * 70)

    assert CLASS_MAP_FILE.exists(), f"Class map file not found: {CLASS_MAP_FILE}"

    class_map = {}
    with open(CLASS_MAP_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                name, val = parts[0], int(parts[1])
                class_map[name] = val

    print(f"  Parsed class map ({len(class_map)} classes):")
    for name, val in sorted(class_map.items(), key=lambda x: x[1]):
        print(f"    {val}: {name}")

    # Verify required classes exist
    for required in ["Car", "Ground", "Road_markings"]:
        assert required in class_map, (
            f"Required class '{required}' not found in class map. "
            f"Available: {list(class_map.keys())}"
        )
    qc_pass("Required classes found", "Car, Ground, Road_markings")

    # Verify all values are non-negative integers
    for name, val in class_map.items():
        assert isinstance(val, int) and val >= 0, (
            f"Class '{name}' has invalid label value: {val}"
        )
    qc_pass("Label values valid", "all non-negative integers")

    return class_map


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Load All Fields and Field Audit
# ──────────────────────────────────────────────────────────────────────────────
def step3_load_and_audit(ply, class_map):
    print("\n" + "=" * 70)
    print("STEP 3: Load All Fields and Field Audit")
    print("=" * 70)

    vertex = ply["vertex"]
    N = vertex.count

    # Extract coordinates (float64)
    x = vertex["x"]
    y = vertex["y"]
    z = vertex["z"]

    # Extract colors
    red = vertex["red"]
    green = vertex["green"]
    blue = vertex["blue"]

    # Extract retained scalar fields
    intensity = vertex["scalar_Intensity"]
    raw_labels_float = vertex["scalar_Label"]

    # Document dropped fields
    dropped_fields = {
        "scalar_GPSTime": "Temporal ordering of original MLS capture; not relevant to simulated VLP-16 sweep",
        "scalar_ScanAngleRank": "Original scanner beam angle; not applicable to simulated VLP-16 geometry",
    }
    print(f"  Dropped fields:")
    for field, reason in dropped_fields.items():
        print(f"    {field}: {reason}")

    # NaN/Inf audit
    print("  NaN/Inf audit:")
    for name, arr in [("x", x), ("y", y), ("z", z),
                      ("intensity", intensity), ("label_raw", raw_labels_float)]:
        has_nan = np.any(np.isnan(arr))
        has_inf = np.any(np.isinf(arr))
        assert not has_nan, f"NaN found in {name}"
        assert not has_inf, f"Inf found in {name}"
        print(f"    {name}: OK (no NaN/Inf)")
    qc_pass("NaN/Inf audit", "all retained numeric fields clean")

    # Label integrity check
    max_frac_error = float(np.max(np.abs(raw_labels_float - np.round(raw_labels_float))))
    print(f"  Label integer-likeness: max fractional error = {max_frac_error:.2e}")
    assert max_frac_error < 1e-5, (
        f"Labels not integer-like: max fractional error = {max_frac_error}"
    )
    qc_pass("Label integer-likeness", f"max error = {max_frac_error:.2e}")

    labels = np.round(raw_labels_float).astype(np.int32)

    # Verify labels are within the class map
    unique_labels = set(np.unique(labels).tolist())
    valid_labels = set(class_map.values())
    assert unique_labels.issubset(valid_labels), (
        f"Unexpected labels: {unique_labels - valid_labels}"
    )
    qc_pass("Label range", f"unique labels {sorted(unique_labels)} ⊆ {sorted(valid_labels)}")

    # Label histogram
    inv_class_map = {v: k for k, v in class_map.items()}
    label_hist_original = {}
    print(f"  Label histogram (N={N:,}):")
    for lbl in sorted(unique_labels):
        cnt = int(np.sum(labels == lbl))
        name = inv_class_map.get(lbl, f"Unknown({lbl})")
        label_hist_original[name] = cnt
        print(f"    {lbl} ({name}): {cnt:,} ({100 * cnt / N:.1f}%)")

    # Assert at least 1 car point
    LABEL_CAR = class_map["Car"]
    car_count = int(np.sum(labels == LABEL_CAR))
    assert car_count > 0, "No car-labeled points found"
    qc_pass("Car points present", f"{car_count:,} points")

    # Verify dtypes
    assert x.dtype == np.float64, f"x dtype should be float64, got {x.dtype}"
    assert y.dtype == np.float64, f"y dtype should be float64, got {y.dtype}"
    assert z.dtype == np.float64, f"z dtype should be float64, got {z.dtype}"
    qc_pass("Coordinate dtype", "float64")

    # Create provenance array
    global_idx = np.arange(N, dtype=np.int64)
    assert global_idx.shape == (N,)
    print(f"  Created global_idx provenance array: shape={global_idx.shape}, dtype={global_idx.dtype}")

    # Stack arrays
    xyz = np.column_stack([x, y, z])  # float64
    rgb = np.column_stack([red, green, blue])  # uint8

    return xyz, rgb, intensity, labels, global_idx, label_hist_original, dropped_fields


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Early Coarse ROI
# ──────────────────────────────────────────────────────────────────────────────
def step4_coarse_roi(xyz, rgb, intensity, labels, global_idx, class_map):
    print("\n" + "=" * 70)
    print("STEP 4: Early Coarse ROI")
    print("=" * 70)

    N_before = xyz.shape[0]

    # Determine road direction (axis with larger extent)
    x_extent = xyz[:, 0].max() - xyz[:, 0].min()
    y_extent = xyz[:, 1].max() - xyz[:, 1].min()
    road_axis = 0 if x_extent > y_extent else 1
    road_axis_name = "X" if road_axis == 0 else "Y"
    road_extent = max(x_extent, y_extent)
    print(f"  Full dataset extents: X={x_extent:.1f}m, Y={y_extent:.1f}m")
    print(f"  Road direction: {road_axis_name} axis (extent={road_extent:.1f}m)")

    # Center of the road axis
    road_center = (xyz[:, road_axis].max() + xyz[:, road_axis].min()) / 2.0

    # Apply ROI: center ± ROI_HALF_EXTENT along road, full perpendicular extent
    roi_mask = (
        (xyz[:, road_axis] >= road_center - ROI_HALF_EXTENT) &
        (xyz[:, road_axis] <= road_center + ROI_HALF_EXTENT)
    )

    xyz = xyz[roi_mask]
    rgb = rgb[roi_mask]
    intensity = intensity[roi_mask]
    labels = labels[roi_mask]
    global_idx = global_idx[roi_mask]

    N_after = xyz.shape[0]
    pct_retained = 100.0 * N_after / N_before
    print(f"  Points: {N_before:,} → {N_after:,} ({pct_retained:.1f}% retained)")

    # Warnings (not assertions)
    if pct_retained < 50:
        qc_warn("ROI retention", f"{pct_retained:.1f}% retained (< 50%)")
    else:
        qc_pass("ROI retention", f"{pct_retained:.1f}% retained")

    roi_extent = xyz[:, road_axis].max() - xyz[:, road_axis].min()
    if roi_extent < 50 or roi_extent > 500:
        qc_warn("ROI extent", f"{roi_extent:.1f}m (outside [50, 500]m heuristic)")
    else:
        qc_pass("ROI extent", f"{roi_extent:.1f}m along {road_axis_name}")

    # Assert at least 1 car point remains
    LABEL_CAR = class_map["Car"]
    car_count_roi = int(np.sum(labels == LABEL_CAR))
    assert car_count_roi > 0, "No car points remain after ROI crop"
    qc_pass("Car points in ROI", f"{car_count_roi:,}")

    # Compute label histogram after ROI
    inv_class_map = {v: k for k, v in class_map.items()}
    label_hist_roi = {}
    for lbl in sorted(np.unique(labels)):
        cnt = int(np.sum(labels == lbl))
        name = inv_class_map.get(lbl, f"Unknown({lbl})")
        label_hist_roi[name] = cnt

    return xyz, rgb, intensity, labels, global_idx, road_axis, road_axis_name, label_hist_roi


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Coordinate Normalization
# ──────────────────────────────────────────────────────────────────────────────
def step5_normalize(xyz):
    print("\n" + "=" * 70)
    print("STEP 5: Coordinate Normalization")
    print("=" * 70)

    # Compute centroid in float64
    scene_origin = xyz.mean(axis=0)  # float64
    print(f"  Scene origin (UTM): [{scene_origin[0]:.6f}, {scene_origin[1]:.6f}, {scene_origin[2]:.6f}]")

    # Recenter (stays float64)
    xyz_centered = xyz - scene_origin

    # Validate
    mean_centered = xyz_centered.mean(axis=0)
    for i, axis in enumerate(["X", "Y", "Z"]):
        assert abs(mean_centered[i]) < 0.01, (
            f"Recentered {axis} mean = {mean_centered[i]:.6f}, not near zero"
        )
    qc_pass("Recenter mean ≈ 0", f"max deviation = {np.abs(mean_centered).max():.2e}m")

    extents = xyz_centered.max(axis=0) - xyz_centered.min(axis=0)
    print(f"  Scene extents: X={extents[0]:.1f}m, Y={extents[1]:.1f}m, Z={extents[2]:.1f}m")

    longest = max(extents[0], extents[1])
    if longest < 50 or longest > 500:
        qc_warn("Scene extent", f"longest horizontal = {longest:.1f}m (outside [50, 500]m)")
    else:
        qc_pass("Scene extent", f"longest horizontal = {longest:.1f}m")

    return xyz_centered, scene_origin


# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Local Ground Plane (RANSAC)
# ──────────────────────────────────────────────────────────────────────────────
def step6_ground_plane(xyz, labels, class_map):
    print("\n" + "=" * 70)
    print("STEP 6: Local Ground Plane (RANSAC)")
    print("=" * 70)

    LABEL_GROUND = class_map["Ground"]
    LABEL_ROAD_MARKINGS = class_map["Road_markings"]

    ground_mask = np.isin(labels, [LABEL_GROUND, LABEL_ROAD_MARKINGS])
    ground_xyz = xyz[ground_mask]
    print(f"  Ground + Road_markings points (full): {ground_xyz.shape[0]:,}")

    # Fit locally near scene center to handle road slope/curvature
    # Use points within ±50m of the XY center for a locally valid plane
    LOCAL_RANSAC_RADIUS = 50.0
    center_xy = ground_xyz[:, :2].mean(axis=0)
    dist_from_center = np.linalg.norm(ground_xyz[:, :2] - center_xy, axis=1)
    local_mask = dist_from_center < LOCAL_RANSAC_RADIUS
    ground_xyz = ground_xyz[local_mask]
    print(f"  Ground points within {LOCAL_RANSAC_RADIUS}m of center: {ground_xyz.shape[0]:,}")

    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(ground_xyz)

    # RANSAC with primary threshold
    ransac_threshold_used = RANSAC_DIST_THRESHOLD
    plane_model, inliers = ground_pcd.segment_plane(
        distance_threshold=ransac_threshold_used,
        ransac_n=RANSAC_N,
        num_iterations=RANSAC_ITERATIONS,
    )
    a, b, c, d = plane_model
    inlier_ratio = len(inliers) / ground_xyz.shape[0]

    # Fallback cascade if inlier ratio too low
    # Ground label includes sidewalks/curbs that deviate from the road plane,
    # so 60% is a realistic minimum for a single-plane fit.
    MIN_INLIER_RATIO = 0.60
    if inlier_ratio < 0.70:
        qc_warn("RANSAC inlier ratio (first attempt)", f"{inlier_ratio:.1%}")
        ransac_threshold_used = 0.10
        print(f"  Retrying with distance_threshold={ransac_threshold_used}m, 2000 iterations...")
        plane_model, inliers = ground_pcd.segment_plane(
            distance_threshold=ransac_threshold_used,
            ransac_n=RANSAC_N,
            num_iterations=2000,
        )
        a, b, c, d = plane_model
        inlier_ratio = len(inliers) / ground_xyz.shape[0]
        if inlier_ratio < MIN_INLIER_RATIO:
            qc_fail("RANSAC inlier ratio (retry)", f"{inlier_ratio:.1%}")
            raise RuntimeError(
                f"Ground RANSAC failed: inlier ratio {inlier_ratio:.1%} < {MIN_INLIER_RATIO:.0%} "
                "even with relaxed threshold"
            )

    # Normalize normal to point upward
    if c < 0:
        a, b, c, d = -a, -b, -c, -d

    normal = np.array([a, b, c], dtype=np.float64)
    normal_unit = normal / np.linalg.norm(normal)

    # Ground Z at origin
    ground_z_at_origin = -d / c

    # Tilt from horizontal
    tilt_rad = np.arccos(np.clip(abs(normal_unit[2]), -1, 1))
    tilt_deg = float(np.degrees(tilt_rad))

    print(f"  Plane equation: {a:.8f}x + {b:.8f}y + {c:.8f}z + {d:.8f} = 0")
    print(f"  Normal (unit): [{normal_unit[0]:.6f}, {normal_unit[1]:.6f}, {normal_unit[2]:.6f}]")
    print(f"  Ground Z at origin: {ground_z_at_origin:.3f}m")
    print(f"  Tilt from horizontal: {tilt_deg:.2f} degrees")
    print(f"  RANSAC inlier ratio: {inlier_ratio:.1%}")

    assert tilt_deg < 5.0, f"Ground tilt {tilt_deg:.1f}° too large"
    qc_pass("Ground tilt", f"{tilt_deg:.2f}°")
    qc_pass("RANSAC inlier ratio", f"{inlier_ratio:.1%}")

    # Verify ground Z is within range of ground-labeled points
    ground_z_min = ground_xyz[:, 2].min()
    ground_z_max = ground_xyz[:, 2].max()
    ground_z_median = float(np.median(ground_xyz[:, 2]))
    z_scene_min = xyz[:, 2].min()
    z_scene_max = xyz[:, 2].max()
    # Ground Z at origin should be within the range of ground point elevations
    assert ground_z_min - 1.0 < ground_z_at_origin < ground_z_max + 1.0, (
        f"Ground Z at origin {ground_z_at_origin:.2f} outside ground point range "
        f"[{ground_z_min:.2f}, {ground_z_max:.2f}]"
    )
    qc_pass("Ground Z position", (
        f"{ground_z_at_origin:.3f}m (ground points Z: [{ground_z_min:.1f}, {ground_z_max:.1f}], "
        f"median={ground_z_median:.1f}, scene Z: [{z_scene_min:.1f}, {z_scene_max:.1f}])"
    ))

    ground_plane = {
        "a": float(a), "b": float(b), "c": float(c), "d": float(d),
        "equation": f"{a:.8f}*x + {b:.8f}*y + {c:.8f}*z + {d:.8f} = 0",
        "normal_unit": normal_unit.tolist(),
        "z_at_origin": float(ground_z_at_origin),
        "tilt_degrees": tilt_deg,
        "inlier_ratio": float(inlier_ratio),
        "ransac_distance_threshold_used": float(ransac_threshold_used),
    }

    return ground_plane, normal_unit


# ──────────────────────────────────────────────────────────────────────────────
# Step 7: Rigorous Car Extraction
# ──────────────────────────────────────────────────────────────────────────────
def step7_car_extraction(xyz, rgb, intensity, labels, global_idx,
                         class_map, ground_plane, n_ground, road_axis):
    print("\n" + "=" * 70)
    print("STEP 7: Rigorous Car Extraction")
    print("=" * 70)

    LABEL_CAR = class_map["Car"]

    # 7a: Filter and cluster
    car_mask = (labels == LABEL_CAR)
    car_xyz = xyz[car_mask]
    car_rgb = rgb[car_mask]
    car_intensity = intensity[car_mask]
    car_global_idx = global_idx[car_mask]
    print(f"  Car points: {car_xyz.shape[0]:,}")

    car_pcd = o3d.geometry.PointCloud()
    car_pcd.points = o3d.utility.Vector3dVector(car_xyz)

    print("  Running DBSCAN clustering...")
    cluster_labels = np.array(
        car_pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=False)
    )

    n_clusters = int(cluster_labels.max() + 1) if len(cluster_labels) > 0 else 0
    noise_count = int(np.sum(cluster_labels == -1))
    print(f"  Found {n_clusters} clusters, {noise_count} noise points")

    if n_clusters == 0:
        # Fallback: reduce min_points
        for fallback_min in [50, 20]:
            print(f"  Retrying DBSCAN with min_points={fallback_min}...")
            cluster_labels = np.array(
                car_pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=fallback_min, print_progress=False)
            )
            n_clusters = int(cluster_labels.max() + 1)
            if n_clusters > 0:
                break
        if n_clusters == 0:
            qc_fail("Car clusters", "No clusters found even with relaxed min_points")
            raise RuntimeError("No car clusters found")

    # Analyze each cluster
    all_clusters = []
    for i in range(n_clusters):
        mask = (cluster_labels == i)
        pts = car_xyz[mask]
        n_pts = int(pts.shape[0])
        centroid = pts.mean(axis=0)
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        dims = bbox_max - bbox_min
        dims_sorted = np.sort(dims)[::-1]  # largest to smallest

        info = {
            "id": i,
            "n_points": n_pts,
            "centroid": centroid.tolist(),
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "dimensions_raw": dims.tolist(),
            "dimensions_sorted": dims_sorted.tolist(),
        }
        all_clusters.append(info)
        print(f"    Cluster {i}: {n_pts:,} pts, "
              f"dims={dims_sorted[0]:.2f}x{dims_sorted[1]:.2f}x{dims_sorted[2]:.2f}m")

    qc_pass("DBSCAN clustering", f"{n_clusters} clusters found")

    # 7b: Select best cluster
    def is_plausible_car(dims_sorted):
        l, w, h = dims_sorted
        return (CAR_LENGTH_RANGE[0] <= l <= CAR_LENGTH_RANGE[1] and
                CAR_WIDTH_RANGE[0] <= w <= CAR_WIDTH_RANGE[1] and
                CAR_HEIGHT_RANGE[0] <= h <= CAR_HEIGHT_RANGE[1])

    plausible = [c for c in all_clusters if is_plausible_car(c["dimensions_sorted"])]

    if not plausible:
        qc_warn("Car dimension filter", "No cluster matches plausible car dimensions; using largest")
        plausible = sorted(all_clusters, key=lambda c: c["n_points"], reverse=True)[:1]
    else:
        qc_pass("Car dimension filter", f"{len(plausible)} plausible candidates")

    best_car = max(plausible, key=lambda c: c["n_points"])
    best_id = best_car["id"]
    assert best_car["n_points"] > 500, (
        f"Selected car cluster has only {best_car['n_points']} points"
    )
    qc_pass("Selected car", (
        f"cluster {best_id}: {best_car['n_points']:,} pts, "
        f"dims={best_car['dimensions_sorted'][0]:.2f}x"
        f"{best_car['dimensions_sorted'][1]:.2f}x"
        f"{best_car['dimensions_sorted'][2]:.2f}m"
    ))

    # Extract selected car's points
    selected_mask = (cluster_labels == best_id)
    sel_xyz = car_xyz[selected_mask]
    sel_rgb = car_rgb[selected_mask]
    sel_intensity = car_intensity[selected_mask]
    sel_global_idx = car_global_idx[selected_mask]

    # 7c: PCA orientation
    xy = sel_xyz[:, :2]  # 2D PCA
    xy_centered = xy - xy.mean(axis=0)
    cov = np.cov(xy_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns sorted ascending, so last is largest
    pc1 = eigvecs[:, -1]  # first principal component
    variance_explained = eigvals[-1] / eigvals.sum()

    yaw_raw = float(np.arctan2(pc1[1], pc1[0]))
    print(f"  PCA: variance explained by PC1 = {variance_explained:.1%}")
    print(f"  Raw yaw (before sign resolution): {np.degrees(yaw_raw):.1f}°")

    if variance_explained < 0.60:
        qc_warn("PCA variance", f"{variance_explained:.1%} (< 60%); car may be nearly square")
    else:
        qc_pass("PCA variance", f"{variance_explained:.1%}")

    # Resolve 180° ambiguity: align with road direction
    road_dir = np.zeros(2, dtype=np.float64)
    road_dir[road_axis] = 1.0  # positive direction along road axis
    if np.dot(pc1, road_dir) < 0:
        yaw_raw += np.pi
    # Normalize to [-π, π]
    yaw = float(np.arctan2(np.sin(yaw_raw), np.cos(yaw_raw)))
    assert np.isfinite(yaw), f"Yaw is not finite: {yaw}"
    print(f"  Resolved yaw: {np.degrees(yaw):.1f}°")
    qc_pass("Yaw", f"{np.degrees(yaw):.1f}° (sign resolved via road direction)")

    # 7d: Define car local frame (ground-plane aligned)
    a = ground_plane["a"]
    b = ground_plane["b"]
    c = ground_plane["c"]
    d = ground_plane["d"]

    # Z_local = ground normal (upward)
    Z_local = n_ground.copy()

    # X_local = yaw direction projected onto ground plane
    yaw_vec_3d = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
    X_local = yaw_vec_3d - np.dot(yaw_vec_3d, Z_local) * Z_local
    X_local = X_local / np.linalg.norm(X_local)

    # Y_local = Z × X (right-hand rule)
    Y_local = np.cross(Z_local, X_local)
    Y_local = Y_local / np.linalg.norm(Y_local)

    # Rotation matrix: rows are local axes expressed in map frame
    R = np.stack([X_local, Y_local, Z_local], axis=0)  # 3x3, map→local

    # Origin: bottom-center projected to LOCAL ground plane near the car
    car_centroid_xy = sel_xyz[:, :2].mean(axis=0)

    # Refit a local ground plane near the car using nearby ground points
    # This avoids the scene-center plane drifting at the car's location
    LABEL_GROUND = class_map["Ground"]
    LABEL_ROAD_MARKINGS = class_map["Road_markings"]
    LOCAL_CAR_GROUND_RADIUS = 15.0  # meters around the car
    LOCAL_REFIT_MIN_INLIER_RATIO = 0.50  # reject weak local fits

    ground_mask_local = np.isin(labels, [LABEL_GROUND, LABEL_ROAD_MARKINGS])
    nearby_ground = xyz[ground_mask_local]
    dist_to_car = np.linalg.norm(nearby_ground[:, :2] - car_centroid_xy, axis=1)
    local_ground_mask = dist_to_car < LOCAL_CAR_GROUND_RADIUS
    local_ground_pts = nearby_ground[local_ground_mask]

    local_refit_used = False
    if local_ground_pts.shape[0] > 100:
        local_gpcd = o3d.geometry.PointCloud()
        local_gpcd.points = o3d.utility.Vector3dVector(local_ground_pts)
        local_plane, local_inliers = local_gpcd.segment_plane(
            distance_threshold=0.05, ransac_n=3, num_iterations=500
        )
        la, lb, lc, ld = local_plane
        if lc < 0:
            la, lb, lc, ld = -la, -lb, -lc, -ld
        local_inlier_ratio = len(local_inliers) / local_ground_pts.shape[0]

        if local_inlier_ratio >= LOCAL_REFIT_MIN_INLIER_RATIO:
            z_ground = -(la * car_centroid_xy[0] + lb * car_centroid_xy[1] + ld) / lc

            # Update local frame normal to use car-local ground plane
            local_normal = np.array([la, lb, lc], dtype=np.float64)
            Z_local = local_normal / np.linalg.norm(local_normal)
            # Recompute X and Y local axes with updated Z
            X_local = yaw_vec_3d - np.dot(yaw_vec_3d, Z_local) * Z_local
            X_local = X_local / np.linalg.norm(X_local)
            Y_local = np.cross(Z_local, X_local)
            Y_local = Y_local / np.linalg.norm(Y_local)
            R = np.stack([X_local, Y_local, Z_local], axis=0)
            local_refit_used = True

            print(f"  Local ground refit near car: {local_ground_pts.shape[0]:,} pts, "
                  f"inlier ratio={local_inlier_ratio:.1%}")
            qc_pass("Car local ground refit",
                     f"{local_ground_pts.shape[0]:,} pts within {LOCAL_CAR_GROUND_RADIUS}m, "
                     f"inlier ratio={local_inlier_ratio:.1%}")
        else:
            print(f"  Local ground refit rejected: inlier ratio={local_inlier_ratio:.1%} "
                  f"< {LOCAL_REFIT_MIN_INLIER_RATIO:.0%}; falling back to scene plane")
            qc_warn("Car local ground refit",
                     f"inlier ratio={local_inlier_ratio:.1%} < {LOCAL_REFIT_MIN_INLIER_RATIO:.0%}; "
                     f"using scene plane")
    else:
        print(f"  Local ground refit skipped: only {local_ground_pts.shape[0]} ground pts nearby; "
              f"using scene plane")
        qc_warn("Car local ground refit",
                 f"Only {local_ground_pts.shape[0]} ground pts nearby; using scene plane")

    if not local_refit_used:
        z_ground = -(a * car_centroid_xy[0] + b * car_centroid_xy[1] + d) / c

    origin = np.array([car_centroid_xy[0], car_centroid_xy[1], z_ground], dtype=np.float64)
    print(f"  Car origin (map frame): [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")

    # Validate orthonormality of R (after potential refit)
    RRt = R @ R.T
    I3 = np.eye(3)
    assert np.allclose(RRt, I3, atol=1e-10), (
        f"R is not orthonormal: max deviation = {np.abs(RRt - I3).max():.2e}"
    )
    assert abs(np.linalg.det(R) - 1.0) < 1e-10, (
        f"R determinant = {np.linalg.det(R):.10f}, not +1"
    )
    qc_pass("Rotation matrix", "orthonormal, det=+1")

    # Transform car points to local frame
    car_local_xyz = (R @ (sel_xyz - origin).T).T  # (N, 3)

    # Validate: 1st percentile of Z near 0
    # Real cars have ground clearance: lowest scanned points (tire sidewalls, body panels)
    # are typically 0.10–0.30m above ground. Values > 0.50m indicate placement error.
    z_pct1 = float(np.percentile(car_local_xyz[:, 2], 1))
    z_min_local = float(car_local_xyz[:, 2].min())
    z_max_local = float(car_local_xyz[:, 2].max())
    print(f"  Car local Z: 1st percentile = {z_pct1:.4f}m, "
          f"min = {z_min_local:.4f}m, max = {z_max_local:.4f}m")
    if z_pct1 > 0.50 or z_pct1 < -0.10:
        qc_warn("Car Z percentile", f"1st percentile = {z_pct1:.4f}m (outside [-0.10, 0.50]m)")
    else:
        qc_pass("Car Z percentile", f"1st percentile = {z_pct1:.4f}m (realistic ground clearance)")

    # 7e: Compute transforms
    # T_map_car0: transforms from car local frame to map frame
    # p_map = R^T @ p_local + origin
    T_map_car0 = np.eye(4, dtype=np.float64)
    T_map_car0[:3, :3] = R.T  # local→map rotation
    T_map_car0[:3, 3] = origin

    T_car0_map = np.linalg.inv(T_map_car0)

    # Validate transforms
    product = T_map_car0 @ T_car0_map
    assert np.allclose(product, np.eye(4), atol=1e-8), (
        f"T_map_car0 @ T_car0_map not identity: max deviation = "
        f"{np.abs(product - np.eye(4)).max():.2e}"
    )
    qc_pass("Transform inverse", f"max deviation = {np.abs(product - np.eye(4)).max():.2e}")

    # Compile car info
    car_info = {
        "cluster_id": best_id,
        "n_points": best_car["n_points"],
        "dimensions": {
            "length": float(best_car["dimensions_sorted"][0]),
            "width": float(best_car["dimensions_sorted"][1]),
            "height": float(best_car["dimensions_sorted"][2]),
        },
        "yaw_rad": yaw,
        "yaw_deg": float(np.degrees(yaw)),
        "pca_variance_explained": float(variance_explained),
        "pca_sign_resolution": "road-direction dot product",
        "centroid_scene": sel_xyz.mean(axis=0).tolist(),
        "origin_scene": origin.tolist(),
        "bottom_z_scene": float(sel_xyz[:, 2].min()),
        "ground_z_at_car": float(z_ground),
        "z_1st_percentile_local": z_pct1,
        "T_map_car0": T_map_car0.tolist(),
        "T_car0_map": T_car0_map.tolist(),
    }

    return (car_local_xyz, sel_rgb, sel_intensity, sel_global_idx,
            car_info, all_clusters)


# ──────────────────────────────────────────────────────────────────────────────
# Step 8: Static Scene Construction by Provenance
# ──────────────────────────────────────────────────────────────────────────────
def step8_static_scene(xyz, rgb, intensity, labels, global_idx,
                       car_global_idx):
    print("\n" + "=" * 70)
    print("STEP 8: Static Scene Construction by Provenance")
    print("=" * 70)

    N_before = xyz.shape[0]
    n_car = car_global_idx.shape[0]

    # Build mask: True for points to keep
    car_idx_set = set(car_global_idx.tolist())
    keep_mask = np.array([gid not in car_idx_set for gid in global_idx], dtype=bool)

    xyz_static = xyz[keep_mask]
    rgb_static = rgb[keep_mask]
    intensity_static = intensity[keep_mask]
    labels_static = labels[keep_mask]
    global_idx_static = global_idx[keep_mask]

    N_after = xyz_static.shape[0]
    expected = N_before - n_car
    assert N_after == expected, (
        f"Static count {N_after} != expected {expected} "
        f"(before={N_before} - car={n_car})"
    )
    qc_pass("Static scene count", f"{N_before:,} - {n_car:,} = {N_after:,}")

    # Verify no car global_idx remain
    remaining_car = set(global_idx_static.tolist()) & car_idx_set
    assert len(remaining_car) == 0, (
        f"{len(remaining_car)} car indices remain in static scene"
    )
    qc_pass("Provenance exclusion", "zero car indices in static scene")

    print(f"  Removed {n_car:,} points (selected car)")
    print(f"  Static scene: {N_after:,} points")

    return xyz_static, rgb_static, intensity_static, labels_static, global_idx_static


# ──────────────────────────────────────────────────────────────────────────────
# Step 9: Statistical Outlier Removal (Optional)
# ──────────────────────────────────────────────────────────────────────────────
def step9_outlier_removal(xyz, rgb, intensity, labels, global_idx, class_map,
                          skip=False):
    print("\n" + "=" * 70)
    print("STEP 9: Statistical Outlier Removal" + (" [SKIPPED]" if skip else ""))
    print("=" * 70)

    label_hist_before = {}
    inv_map = {v: k for k, v in class_map.items()}
    for lbl in np.unique(labels):
        label_hist_before[inv_map.get(int(lbl), str(lbl))] = int(np.sum(labels == lbl))

    if skip:
        qc_pass("Outlier removal", "SKIPPED by user flag")
        return xyz, rgb, intensity, labels, global_idx, label_hist_before, None

    N_before = xyz.shape[0]
    print(f"  Building point cloud ({N_before:,} points)...")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    print(f"  Running outlier removal (nb_neighbors={OUTLIER_NB_NEIGHBORS}, "
          f"std_ratio={OUTLIER_STD_RATIO})...")
    t0 = time.time()
    _, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    inlier_idx = np.array(inlier_idx)
    N_after = len(inlier_idx)
    n_removed = N_before - N_after
    removal_pct = 100.0 * n_removed / N_before
    print(f"  Before: {N_before:,}, After: {N_after:,}, "
          f"Removed: {n_removed:,} ({removal_pct:.2f}%)")

    if removal_pct > 5.0:
        qc_fail("Outlier removal %", f"{removal_pct:.2f}% (> 5%)")
        raise RuntimeError(f"Outlier removal too aggressive: {removal_pct:.1f}%")
    elif removal_pct > 2.0:
        qc_warn("Outlier removal %", f"{removal_pct:.2f}% (> 2%)")
    else:
        qc_pass("Outlier removal %", f"{removal_pct:.2f}%")

    # Apply to all arrays
    xyz = xyz[inlier_idx]
    rgb = rgb[inlier_idx]
    intensity = intensity[inlier_idx]
    labels = labels[inlier_idx]
    global_idx = global_idx[inlier_idx]

    # Check thin structure survival
    LABEL_UTILITY = class_map["Utility_line"]
    LABEL_POLE = class_map["Pole"]
    for lname, lval in [("Utility_line", LABEL_UTILITY), ("Pole", LABEL_POLE)]:
        before = label_hist_before.get(lname, 0)
        after = int(np.sum(labels == lval))
        if before > 0:
            survival = 100.0 * after / before
            print(f"  {lname}: {before:,} → {after:,} ({survival:.1f}% survived)")
            if survival < 80:
                qc_warn(f"{lname} survival", f"{survival:.1f}% (< 80%)")
            else:
                qc_pass(f"{lname} survival", f"{survival:.1f}%")

    # Label histogram after
    label_hist_after = {}
    for lbl in np.unique(labels):
        label_hist_after[inv_map.get(int(lbl), str(lbl))] = int(np.sum(labels == lbl))

    return xyz, rgb, intensity, labels, global_idx, label_hist_before, label_hist_after


# ──────────────────────────────────────────────────────────────────────────────
# Step 10: Voxel Downsampling with Attribute Aggregation
# ──────────────────────────────────────────────────────────────────────────────
def step10_voxel_downsample(xyz, rgb, intensity, labels, global_idx, class_map):
    print("\n" + "=" * 70)
    print("STEP 10: Voxel Downsampling with Attribute Aggregation")
    print("=" * 70)

    N_before = xyz.shape[0]
    inv_map = {v: k for k, v in class_map.items()}

    # Label histogram before
    label_hist_before_ds = {}
    for lbl in np.unique(labels):
        label_hist_before_ds[inv_map.get(int(lbl), str(lbl))] = int(np.sum(labels == lbl))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)

    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()

    print(f"  Voxel size: {VOXEL_SIZE}m")
    print(f"  Input points: {N_before:,}")

    downsampled_pcd, _, trace_indices = pcd.voxel_down_sample_and_trace(
        voxel_size=VOXEL_SIZE, min_bound=min_bound, max_bound=max_bound
    )

    N_after = len(downsampled_pcd.points)
    reduction = N_before / N_after if N_after > 0 else float("inf")
    print(f"  Output points: {N_after:,} (reduction: {reduction:.1f}x)")

    assert 500_000 < N_after < 5_000_000, (
        f"Downsampled count {N_after:,} outside expected [500K, 5M]"
    )
    qc_pass("Downsample count", f"{N_after:,} ({reduction:.1f}x reduction)")

    # Aggregate labels (majority vote) and intensity (mean)
    print("  Aggregating labels and intensity per voxel...")
    ds_labels = np.zeros(N_after, dtype=np.int32)
    ds_intensity = np.zeros(N_after, dtype=np.float32)

    # Build flat trace representation: offsets + flat array
    # This is much more compact than storing each voxel's trace as a separate array
    trace_flat_parts = []
    trace_offsets = np.zeros(N_after + 1, dtype=np.int64)
    for i, indices in enumerate(trace_indices):
        idx = np.array(indices, dtype=np.int64)
        if len(idx) > 0:
            # Majority vote for label
            votes = labels[idx]
            ds_labels[i] = int(np.bincount(votes.astype(np.int64)).argmax())
            # Mean intensity
            ds_intensity[i] = float(intensity[idx].mean())
            trace_flat_parts.append(idx)
        trace_offsets[i + 1] = trace_offsets[i] + len(idx)
    trace_flat = np.concatenate(trace_flat_parts) if trace_flat_parts else np.array([], dtype=np.int64)
    trace_data = {"offsets": trace_offsets, "indices": trace_flat}

    # Label histogram after downsampling
    label_hist_after_ds = {}
    for lbl in np.unique(ds_labels):
        label_hist_after_ds[inv_map.get(int(lbl), str(lbl))] = int(np.sum(ds_labels == lbl))

    print(f"  Label histogram after downsampling:")
    for name, cnt in sorted(label_hist_after_ds.items(), key=lambda x: -x[1]):
        before_cnt = label_hist_before_ds.get(name, 0)
        print(f"    {name}: {before_cnt:,} → {cnt:,}")

    # Check thin structures
    for lname in ["Utility_line", "Pole"]:
        before = label_hist_before_ds.get(lname, 0)
        after = label_hist_after_ds.get(lname, 0)
        if before > 0:
            survival = 100.0 * after / before
            if survival < 50:
                qc_warn(f"{lname} downsample survival", f"{survival:.1f}% (< 50%)")
            else:
                qc_pass(f"{lname} downsample survival", f"{survival:.1f}%")

    return (downsampled_pcd, ds_labels, ds_intensity, trace_data,
            label_hist_before_ds, label_hist_after_ds)


# ──────────────────────────────────────────────────────────────────────────────
# Step 11: Scene Coverage Verification
# ──────────────────────────────────────────────────────────────────────────────
def step11_coverage(downsampled_pcd, road_axis_name):
    print("\n" + "=" * 70)
    print("STEP 11: Scene Coverage Verification")
    print("=" * 70)

    pts = np.asarray(downsampled_pcd.points)
    scene_min = pts.min(axis=0)
    scene_max = pts.max(axis=0)
    extent = scene_max - scene_min

    print(f"  Scene bounding box:")
    print(f"    X: [{scene_min[0]:.1f}, {scene_max[0]:.1f}] ({extent[0]:.1f}m)")
    print(f"    Y: [{scene_min[1]:.1f}, {scene_max[1]:.1f}] ({extent[1]:.1f}m)")
    print(f"    Z: [{scene_min[2]:.1f}, {scene_max[2]:.1f}] ({extent[2]:.1f}m)")

    road_idx = 0 if road_axis_name == "X" else 1
    road_extent = extent[road_idx]
    print(f"  Road direction: {road_axis_name} ({road_extent:.1f}m)")

    lidar_range = 100.0
    usable = road_extent - 2 * lidar_range
    print(f"  Usable trajectory (VLP-16 range={lidar_range}m): {usable:.1f}m")

    if usable <= 0:
        qc_warn("Scene coverage", f"Scene ({road_extent:.0f}m) < 2×LiDAR range")
        for rr in [75, 50, 30]:
            u = road_extent - 2 * rr
            if u > 0:
                speed = u / 15.0
                print(f"    At {rr}m range: {u:.0f}m usable, "
                      f"max speed = {speed:.1f} m/s ({speed * 3.6:.0f} km/h)")
    else:
        max_speed = usable / 15.0
        print(f"  Max speed for 15s: {max_speed:.1f} m/s ({max_speed * 3.6:.0f} km/h)")
        qc_pass("Scene coverage", f"{usable:.0f}m usable at 100m range")

    coverage_info = {
        "bounding_box_min": scene_min.tolist(),
        "bounding_box_max": scene_max.tolist(),
        "extent_m": extent.tolist(),
        "road_direction": road_axis_name,
        "road_extent_m": float(road_extent),
    }
    return coverage_info


# ──────────────────────────────────────────────────────────────────────────────
# Step 12: Save All Outputs
# ──────────────────────────────────────────────────────────────────────────────
def step12_save(downsampled_pcd, ds_labels, ds_intensity, trace_data,
                car_local_xyz, car_rgb, car_intensity,
                metadata, checksum):
    print("\n" + "=" * 70)
    print("STEP 12: Save All Outputs and Reload Verification")
    print("=" * 70)

    # 1. Scene static PLY
    scene_path = OUT_DIR / "scene_static.ply"
    o3d.io.write_point_cloud(str(scene_path), downsampled_pcd, write_ascii=False)
    print(f"  Saved: {scene_path} ({len(downsampled_pcd.points):,} points)")

    # 2. Scene labels
    labels_path = OUT_DIR / "scene_labels.npy"
    np.save(str(labels_path), ds_labels)
    print(f"  Saved: {labels_path}")

    # 3. Scene intensity
    intensity_path = OUT_DIR / "scene_intensity.npy"
    np.save(str(intensity_path), ds_intensity)
    print(f"  Saved: {intensity_path}")

    # 4. Voxel trace (compressed, separate from metadata)
    trace_path = OUT_DIR / "voxel_trace.npz"
    np.savez_compressed(str(trace_path), **trace_data)
    print(f"  Saved: {trace_path}")

    # 5. Car local PLY
    car_pcd = o3d.geometry.PointCloud()
    car_pcd.points = o3d.utility.Vector3dVector(car_local_xyz)
    car_pcd.colors = o3d.utility.Vector3dVector(car_rgb.astype(np.float64) / 255.0)
    car_path = OUT_DIR / "car_local.ply"
    o3d.io.write_point_cloud(str(car_path), car_pcd, write_ascii=False)
    print(f"  Saved: {car_path} ({car_local_xyz.shape[0]:,} points)")

    # 6. Car intensity
    car_int_path = OUT_DIR / "car_intensity.npy"
    np.save(str(car_int_path), car_intensity)
    print(f"  Saved: {car_int_path}")

    # 7. Metadata (initial write — will be overwritten after reload verification
    #    to capture the complete qc_gates list)
    meta_path = OUT_DIR / "metadata.json"
    with open(str(meta_path), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {meta_path}")

    # ── Reload verification ──
    print("\n  Reload verification:")

    # Verify scene PLY
    reload_pcd = o3d.io.read_point_cloud(str(scene_path))
    n_reload = len(reload_pcd.points)
    n_expected = len(downsampled_pcd.points)
    assert n_reload == n_expected, (
        f"Scene PLY reload: {n_reload} != {n_expected}"
    )
    qc_pass("Scene PLY reload", f"{n_reload:,} points")

    # Verify labels
    reload_labels = np.load(str(labels_path))
    assert reload_labels.shape == ds_labels.shape, "Labels shape mismatch"
    assert np.array_equal(reload_labels, ds_labels), "Labels content mismatch"
    qc_pass("Labels reload", f"shape={reload_labels.shape}")

    # Verify intensity
    reload_int = np.load(str(intensity_path))
    assert reload_int.shape == ds_intensity.shape, "Intensity shape mismatch"
    qc_pass("Intensity reload", f"shape={reload_int.shape}")

    # Verify car PLY
    reload_car = o3d.io.read_point_cloud(str(car_path))
    assert len(reload_car.points) == car_local_xyz.shape[0], "Car PLY count mismatch"
    qc_pass("Car PLY reload", f"{len(reload_car.points):,} points")

    # Verify metadata JSON (check keys from the initial write)
    with open(str(meta_path)) as f:
        reload_meta = json.load(f)
    expected_keys = [
        "sha256_checksum", "library_versions", "ply_schema", "dropped_fields",
        "scene_origin", "coordinate_convention", "ground_plane", "car_cluster",
        "static_scene", "all_car_clusters", "parameters_used", "label_histograms",
        "row_order_alignment", "class_map", "qc_gates",
    ]
    for key in expected_keys:
        assert key in reload_meta, f"Missing metadata key: '{key}'"
    qc_pass("Metadata keys", f"all {len(expected_keys)} expected keys present")

    # Verify transform roundtrip
    T1 = np.array(reload_meta["car_cluster"]["T_map_car0"])
    T2 = np.array(reload_meta["car_cluster"]["T_car0_map"])
    product = T1 @ T2
    max_dev = float(np.abs(product - np.eye(4)).max())
    assert max_dev < 1e-6, (
        f"Post-serialization transform roundtrip: max deviation = {max_dev:.2e}"
    )
    qc_pass("Transform roundtrip (post-JSON)", f"max deviation = {max_dev:.2e}")

    # ── Final metadata write with complete qc_gates ──
    with open(str(meta_path), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Re-saved metadata.json with complete qc_gates ({len(metadata['qc_gates'])} gates)")

    # Print file sizes
    print("\n  Output file sizes:")
    for path in [scene_path, labels_path, intensity_path, trace_path,
                 car_path, car_int_path, meta_path]:
        size = path.stat().st_size
        print(f"    {path.name}: {size:,} bytes ({size / 1e6:.1f} MB)")


# ──────────────────────────────────────────────────────────────────────────────
# Step 13: QC Report
# ──────────────────────────────────────────────────────────────────────────────
def step13_qc_report(metadata):
    print("\n" + "=" * 70)
    print("STEP 13: QC Report Generation")
    print("=" * 70)

    lines = []
    lines.append("# Preprocessing QC Report\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Input info
    lines.append("## Input File\n")
    lines.append(f"- **File**: `{RAW_PLY}`")
    lines.append(f"- **Size**: {EXPECTED_FILE_SIZE:,} bytes")
    lines.append(f"- **SHA256**: `{metadata['sha256_checksum']}`")
    lines.append("")

    # Class map
    lines.append("## Class Map (parsed from file)\n")
    lines.append("| Label | Class |")
    lines.append("|-------|-------|")
    for name, val in sorted(metadata["class_map"].items(), key=lambda x: x[1]):
        lines.append(f"| {val} | {name} |")
    lines.append("")

    # Label histograms
    lines.append("## Label Histograms\n")
    for stage_name, hist in metadata["label_histograms"].items():
        lines.append(f"### {stage_name}\n")
        if isinstance(hist, str):
            lines.append(f"_{hist}_\n")
            continue
        lines.append("| Class | Count |")
        lines.append("|-------|-------|")
        for name, cnt in sorted(hist.items(), key=lambda x: -x[1]):
            lines.append(f"| {name} | {cnt:,} |")
        lines.append("")

    # Ground plane
    gp = metadata["ground_plane"]
    lines.append("## Ground Plane\n")
    lines.append(f"- **Equation**: `{gp['equation']}`")
    lines.append(f"- **Normal (unit)**: `{gp['normal_unit']}`")
    lines.append(f"- **Z at origin**: {gp['z_at_origin']:.3f}m")
    lines.append(f"- **Tilt**: {gp['tilt_degrees']:.2f}°")
    lines.append(f"- **Inlier ratio**: {gp['inlier_ratio']:.1%}")
    lines.append("")

    # Car extraction
    car = metadata["car_cluster"]
    lines.append("## Selected Car Cluster\n")
    lines.append(f"- **Cluster ID**: {car['cluster_id']}")
    lines.append(f"- **Points**: {car['n_points']:,}")
    dims = car["dimensions"]
    lines.append(f"- **Dimensions**: {dims['length']:.2f} x {dims['width']:.2f} x {dims['height']:.2f}m (LxWxH)")
    lines.append(f"- **Yaw**: {car['yaw_deg']:.1f}° ({car['yaw_rad']:.4f} rad)")
    lines.append(f"- **PCA variance explained**: {car['pca_variance_explained']:.1%}")
    lines.append(f"- **Sign resolution**: {car['pca_sign_resolution']}")
    lines.append(f"- **Z 1st percentile (local)**: {car['z_1st_percentile_local']:.4f}m")
    lines.append("")

    # T_map_car0
    lines.append("### T_map_car0\n")
    lines.append("```")
    T = np.array(car["T_map_car0"])
    for row in T:
        lines.append("  " + "  ".join(f"{v:12.6f}" for v in row))
    lines.append("```\n")

    # All clusters
    lines.append("### All DBSCAN Clusters\n")
    lines.append("| ID | Points | L (m) | W (m) | H (m) | Selected |")
    lines.append("|----|--------|-------|-------|-------|----------|")
    for cl in metadata["all_car_clusters"]:
        ds = cl["dimensions_sorted"]
        sel = "**YES**" if cl["id"] == car["cluster_id"] else ""
        lines.append(f"| {cl['id']} | {cl['n_points']:,} | {ds[0]:.2f} | {ds[1]:.2f} | {ds[2]:.2f} | {sel} |")
    lines.append("")

    # Static scene
    ss = metadata["static_scene"]
    lines.append("## Static Scene\n")
    for stage, count in ss["point_counts"].items():
        lines.append(f"- **{stage}**: {count:,}")
    lines.append(f"- **Voxel size**: {ss['voxel_size']}m")
    lines.append("")

    # Scene coverage
    cov = metadata.get("scene_coverage", {})
    if cov:
        lines.append("## Scene Coverage\n")
        lines.append(f"- **Road direction**: {cov.get('road_direction', 'N/A')}")
        lines.append(f"- **Road extent**: {cov.get('road_extent_m', 0):.1f}m")
        ext = cov.get("extent_m", [0, 0, 0])
        lines.append(f"- **Full extent**: {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f}m")
        lines.append("")

    # Parameters
    lines.append("## Parameters\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    for k, v in metadata["parameters_used"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # QC Gates
    lines.append("## Validation Gates\n")
    lines.append("| Gate | Status | Detail |")
    lines.append("|------|--------|--------|")
    for gate in qc_gates:
        status = gate["status"]
        emoji = {"PASS": "PASS", "FAIL": "**FAIL**", "WARN": "WARN"}[status]
        lines.append(f"| {gate['name']} | {emoji} | {gate['detail']} |")
    lines.append("")

    # Summary
    n_pass = sum(1 for g in qc_gates if g["status"] == "PASS")
    n_warn = sum(1 for g in qc_gates if g["status"] == "WARN")
    n_fail = sum(1 for g in qc_gates if g["status"] == "FAIL")
    lines.append(f"## Summary\n")
    lines.append(f"- **PASS**: {n_pass}")
    lines.append(f"- **WARN**: {n_warn}")
    lines.append(f"- **FAIL**: {n_fail}")

    report = "\n".join(lines) + "\n"
    report_path = OUT_DIR / "qc_report.md"
    with open(str(report_path), "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")
    print(f"  Gates: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Preprocess Toronto-3D L002")
    parser.add_argument("--skip-outlier-removal", action="store_true",
                        help="Skip statistical outlier removal (Step 9)")
    args = parser.parse_args()

    t_start = time.time()

    # Step 1
    ply, checksum = step1_integrity()

    # Step 2
    class_map = step2_parse_class_map()

    # Step 3
    (xyz, rgb, intensity, labels, global_idx,
     label_hist_original, dropped_fields) = step3_load_and_audit(ply, class_map)
    del ply  # free memory

    # Step 4
    (xyz, rgb, intensity, labels, global_idx,
     road_axis, road_axis_name, label_hist_roi) = step4_coarse_roi(
        xyz, rgb, intensity, labels, global_idx, class_map
    )

    # Step 5
    xyz, scene_origin = step5_normalize(xyz)

    # Step 6
    ground_plane, n_ground = step6_ground_plane(xyz, labels, class_map)

    # Step 7
    (car_local_xyz, car_rgb, car_intensity_arr, car_global_idx,
     car_info, all_clusters) = step7_car_extraction(
        xyz, rgb, intensity, labels, global_idx,
        class_map, ground_plane, n_ground, road_axis
    )

    # Step 8
    n_after_roi = xyz.shape[0]  # capture before car removal
    (xyz_s, rgb_s, int_s, lab_s, gidx_s) = step8_static_scene(
        xyz, rgb, intensity, labels, global_idx, car_global_idx
    )
    n_after_car_removal = xyz_s.shape[0]  # capture after car removal

    # Step 9
    (xyz_s, rgb_s, int_s, lab_s, gidx_s,
     label_hist_pre_outlier, label_hist_post_outlier) = step9_outlier_removal(
        xyz_s, rgb_s, int_s, lab_s, gidx_s, class_map,
        skip=args.skip_outlier_removal
    )
    n_after_outlier = xyz_s.shape[0]  # capture after outlier removal

    # Step 10
    (ds_pcd, ds_labels, ds_intensity, trace_data,
     label_hist_pre_ds, label_hist_post_ds) = step10_voxel_downsample(
        xyz_s, rgb_s, int_s, lab_s, gidx_s, class_map
    )

    # Step 11
    coverage_info = step11_coverage(ds_pcd, road_axis_name)

    # Build metadata
    ds_pts = np.asarray(ds_pcd.points)
    from importlib.metadata import version as _pkg_version
    metadata = {
        "sha256_checksum": checksum,
        "library_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "open3d": o3d.__version__,
            "plyfile": _pkg_version("plyfile"),
            "scipy": _pkg_version("scipy"),
        },
        "ply_schema": {
            "endianness": "binary_little_endian",
            "vertex_count": EXPECTED_VERTEX_COUNT,
            "properties": [{"name": n, "type": t} for n, t in EXPECTED_SCHEMA],
        },
        "dropped_fields": dropped_fields,
        "scene_origin": scene_origin.tolist(),
        "coordinate_convention": {
            "units": "meters",
            "axes_inferred": (
                "X=East, Y=North, Z=Up "
                "(inferred from dataset paper, not directly verified from dataset metadata)"
            ),
            "handedness": "right-handed (inferred)",
            "origin": "centroid of ROI",
            "crs_original": "NAD83 / UTM Zone 17N EPSG:26917 (inferred from dataset paper)",
        },
        "ground_plane": ground_plane,
        "car_cluster": car_info,
        "all_car_clusters": all_clusters,
        "static_scene": {
            "point_counts": {
                "after_roi": int(n_after_roi),
                "after_car_removal": int(n_after_car_removal),
                "after_outlier_removal": int(n_after_outlier),
                "after_voxel_downsample": int(len(ds_pcd.points)),
            },
            "voxel_size": VOXEL_SIZE,
            "bounding_box_min": ds_pts.min(axis=0).tolist(),
            "bounding_box_max": ds_pts.max(axis=0).tolist(),
        },
        "scene_coverage": coverage_info,
        "parameters_used": {
            "voxel_size_m": VOXEL_SIZE,
            "dbscan_eps_m": DBSCAN_EPS,
            "dbscan_min_points": DBSCAN_MIN_POINTS,
            "outlier_nb_neighbors": OUTLIER_NB_NEIGHBORS,
            "outlier_std_ratio": OUTLIER_STD_RATIO,
            "outlier_removal_skipped": args.skip_outlier_removal,
            "ransac_distance_threshold_m": RANSAC_DIST_THRESHOLD,
            "ransac_n": RANSAC_N,
            "ransac_iterations": RANSAC_ITERATIONS,
            "roi_half_extent_m": ROI_HALF_EXTENT,
            "car_length_range_m": list(CAR_LENGTH_RANGE),
            "car_width_range_m": list(CAR_WIDTH_RANGE),
            "car_height_range_m": list(CAR_HEIGHT_RANGE),
        },
        "label_histograms": {
            "original": label_hist_original,
            "after_roi": label_hist_roi,
            "before_outlier_removal": label_hist_pre_outlier,
            "after_outlier_removal": label_hist_post_outlier if label_hist_post_outlier else "SKIPPED",
            "before_voxel_downsample": label_hist_pre_ds,
            "after_voxel_downsample": label_hist_post_ds,
        },
        "row_order_alignment": (
            "scene_static.ply, scene_labels.npy, and scene_intensity.npy share "
            "the same row ordering: row i in all three files corresponds to voxel i. "
            "voxel_trace.npz uses CSR-style storage: voxel i maps to source indices "
            "indices[offsets[i]:offsets[i+1]]."
        ),
        "class_map": class_map,
        "qc_gates": qc_gates,
    }

    # Step 12
    step12_save(ds_pcd, ds_labels, ds_intensity, trace_data,
                car_local_xyz, car_rgb, car_intensity_arr,
                metadata, checksum)

    # Step 13
    step13_qc_report(metadata)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"PREPROCESSING COMPLETE in {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Final summary
    n_fail = sum(1 for g in qc_gates if g["status"] == "FAIL")
    if n_fail > 0:
        print(f"WARNING: {n_fail} QC gate(s) FAILED. Check qc_report.md.")
        sys.exit(1)
    else:
        print("All QC gates passed.")


if __name__ == "__main__":
    main()
