"""MCAP writer: serialise simulation output to MCAP with ROS 2 schemas.

Produces a single MCAP file playable in Foxglove Studio, containing:
  /tf_static  - static transforms (published once at t=0)
  /tf         - dynamic transforms (at tf_rate_hz)
  /velodyne_points - VLP-16 PointCloud2 (at lidar_rate_hz)
  /camera/image_raw/compressed - sensor_msgs/CompressedImage (png/jpeg)
  /camera/camera_info - sensor_msgs/CameraInfo (at camera_rate_hz)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from mcap_ros2.writer import Writer as McapWriter

from .config import CameraIntrinsics, FrameNames
from .ros2_msgdefs import (
    CAMERAINFO_MSGDEF,
    COMPRESSEDIMAGE_MSGDEF,
    PF_FLOAT32,
    PF_UINT16,
    POINTCLOUD2_MSGDEF,
    TFMESSAGE_MSGDEF,
)


# -- Rotation matrix -> quaternion -------------------------------------

def _rotation_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w).

    Uses Shepperd's method for numerical stability.
    Returns ROS convention: (x, y, z, w).
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return (float(x), float(y), float(z), float(w))


# ── Timestamp helpers ─────────────────────────────────────────────────

def _ns_to_stamp(ns: int) -> dict:
    """Convert nanoseconds to ROS2 Time dict."""
    # ROS messages split time into integer seconds + remainder nanoseconds.
    return {"sec": int(ns // 1_000_000_000), "nanosec": int(ns % 1_000_000_000)}


# ── Transform message builders ────────────────────────────────────────

def _make_transform_stamped(
    stamp_ns: int,
    parent_frame: str,
    child_frame: str,
    T: np.ndarray,
) -> dict:
    """Build a geometry_msgs/TransformStamped dict from a 4x4 SE3 matrix.

    T is T_parent_child (transforms points FROM child INTO parent).
    The translation and rotation describe the child frame's pose in the parent.
    """
    # Cast to Python floats so mcap_ros2 serializes plain scalars, not NumPy
    # scalar objects.
    tx, ty, tz = T[0, 3], T[1, 3], T[2, 3]
    qx, qy, qz, qw = _rotation_to_quaternion(T[:3, :3])

    return {
        "header": {
            "stamp": _ns_to_stamp(stamp_ns),
            "frame_id": parent_frame,
        },
        "child_frame_id": child_frame,
        "transform": {
            "translation": {"x": float(tx), "y": float(ty), "z": float(tz)},
            "rotation": {"x": qx, "y": qy, "z": qz, "w": qw},
        },
    }


# ── SimWriter class ───────────────────────────────────────────────────

class SimWriter:
    """Streaming MCAP writer for the sensor simulation.

    Usage:
        with SimWriter("output.mcap", frame_names, intrinsics) as w:
            w.write_tf_static(stamp_ns, tf_snapshot)
            w.write_tf(stamp_ns, tf_snapshot)
            w.write_pointcloud2(stamp_ns, lidar_frame)
            w.write_compressed_image(stamp_ns, camera_frame, fmt="png")
            w.write_camera_info(stamp_ns)
    """

    def __init__(
        self,
        path: str | Path,
        frame_names: FrameNames,
        intrinsics: CameraIntrinsics,
    ):
        self._path = str(path)
        self._fn = frame_names
        self._intrinsics = intrinsics
        self._writer: McapWriter | None = None
        self._schemas: dict[str, object] = {}
        self._seq_tf = 0
        self._seq_lidar = 0
        self._seq_camera = 0

    def __enter__(self):
        self._writer = McapWriter(self._path)
        self._writer.__enter__()
        self._register_schemas()
        return self

    def __exit__(self, *exc):
        if self._writer:
            self._writer.__exit__(*exc)
            self._writer = None

    def _register_schemas(self):
        w = self._writer
        # Register schemas once up front; later writes only send message dicts.
        self._schemas["tf"] = w.register_msgdef(
            "tf2_msgs/msg/TFMessage", TFMESSAGE_MSGDEF
        )
        self._schemas["pc2"] = w.register_msgdef(
            "sensor_msgs/msg/PointCloud2", POINTCLOUD2_MSGDEF
        )
        self._schemas["camera_info"] = w.register_msgdef(
            "sensor_msgs/msg/CameraInfo", CAMERAINFO_MSGDEF
        )
        self._schemas["compressed_image"] = w.register_msgdef(
            "sensor_msgs/msg/CompressedImage", COMPRESSEDIMAGE_MSGDEF
        )

    # ── TF ─────────────────────────────────────────────────────────

    def write_tf_static(self, stamp_ns: int, tf_snapshot) -> None:
        """Write /tf_static with all static transforms."""
        fn = self._fn
        transforms = [
            _make_transform_stamped(
                stamp_ns, fn.ego_body, fn.lidar,
                tf_snapshot.T_base_velodyne,
            ),
            _make_transform_stamped(
                stamp_ns, fn.ego_body, fn.camera_body,
                tf_snapshot.T_base_camera_link,
            ),
            _make_transform_stamped(
                stamp_ns, fn.camera_body, fn.camera_optical,
                tf_snapshot.T_camera_link_optical,
            ),
        ]
        msg = {"transforms": transforms}
        self._writer.write_message(
            topic="/tf_static",
            schema=self._schemas["tf"],
            message=msg,
            log_time=stamp_ns,
            publish_time=stamp_ns,
            sequence=0,
        )

    def write_tf(self, stamp_ns: int, tf_snapshot) -> None:
        """Write /tf with dynamic transforms (map -> base_link)."""
        fn = self._fn
        transforms = [
            _make_transform_stamped(
                stamp_ns, fn.fixed, fn.ego_body,
                tf_snapshot.T_map_base,
            ),
        ]
        # Sort agent frame ids for deterministic MCAP output ordering.
        for child_frame, T_map_agent in sorted(getattr(tf_snapshot, "T_map_agents", {}).items()):
            transforms.append(
                _make_transform_stamped(
                    stamp_ns, fn.fixed, child_frame, T_map_agent,
                )
            )
        msg = {"transforms": transforms}
        self._writer.write_message(
            topic="/tf",
            schema=self._schemas["tf"],
            message=msg,
            log_time=stamp_ns,
            publish_time=stamp_ns,
            sequence=self._seq_tf,
        )
        self._seq_tf += 1

    # ── PointCloud2 ────────────────────────────────────────────────

    def write_pointcloud2(self, stamp_ns: int, lidar_frame) -> None:
        """Write /velodyne_points as sensor_msgs/PointCloud2.

        Locked contract: x(f32), y(f32), z(f32), intensity(f32),
        ring(u16), pad(2 bytes) -> point_step=20.
        """
        fn = self._fn
        n_pts = len(lidar_frame.points)

        # Pack binary data: x,y,z (f32), intensity (f32), ring (u16), pad(2)
        point_step = 20
        dt = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("intensity", "<f4"),
            ("ring", "<u2"),
            ("_pad", "<u2"),
        ])
        # Build an interleaved binary buffer that matches the PointCloud2 field
        # layout exactly (point_step=20).
        structured = np.zeros(n_pts, dtype=dt)
        structured["x"] = lidar_frame.points[:, 0]
        structured["y"] = lidar_frame.points[:, 1]
        structured["z"] = lidar_frame.points[:, 2]
        structured["intensity"] = lidar_frame.intensity
        structured["ring"] = lidar_frame.ring
        # tobytes() materializes the final ROS byte payload once.
        buf = structured.tobytes()

        fields = [
            {"name": "x", "offset": 0, "datatype": PF_FLOAT32, "count": 1},
            {"name": "y", "offset": 4, "datatype": PF_FLOAT32, "count": 1},
            {"name": "z", "offset": 8, "datatype": PF_FLOAT32, "count": 1},
            {"name": "intensity", "offset": 12, "datatype": PF_FLOAT32, "count": 1},
            {"name": "ring", "offset": 16, "datatype": PF_UINT16, "count": 1},
        ]

        msg = {
            "header": {
                "stamp": _ns_to_stamp(stamp_ns),
                "frame_id": fn.lidar,
            },
            "height": 1,
            "width": n_pts,
            "fields": fields,
            "is_bigendian": False,
            "point_step": point_step,
            "row_step": point_step * n_pts,
            "data": buf,
            "is_dense": True,
        }

        self._writer.write_message(
            topic="/velodyne_points",
            schema=self._schemas["pc2"],
            message=msg,
            log_time=stamp_ns,
            publish_time=stamp_ns,
            sequence=self._seq_lidar,
        )
        self._seq_lidar += 1

    # ── CompressedImage ──────────────────────────────────────────────

    def write_compressed_image(
        self, stamp_ns: int, camera_frame, fmt: str = "png"
    ) -> None:
        """Write /camera/image_raw/compressed as sensor_msgs/CompressedImage.

        Args:
            stamp_ns: Timestamp in nanoseconds.
            camera_frame: CameraFrame with .image (H, W, 3) uint8 RGB.
            fmt: "png" or "jpeg".
        """
        fn = self._fn
        # cv2 expects BGR, but the renderer stores RGB.
        bgr = cv2.cvtColor(camera_frame.image, cv2.COLOR_RGB2BGR)

        if fmt == "png":
            ok, buf = cv2.imencode(".png", bgr)
            format_str = "png"
        elif fmt == "jpeg":
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            format_str = "jpeg"
        else:
            raise ValueError(f"Unsupported compression format: {fmt}")

        # A failed encode would write an invalid MCAP message with garbage bytes.
        if not ok:
            raise RuntimeError(f"cv2.imencode failed for format {fmt}")

        msg = {
            "header": {
                "stamp": _ns_to_stamp(stamp_ns),
                "frame_id": fn.camera_optical,
            },
            "format": format_str,
            "data": buf.tobytes(),
        }

        self._writer.write_message(
            topic="/camera/image_raw/compressed",
            schema=self._schemas["compressed_image"],
            message=msg,
            log_time=stamp_ns,
            publish_time=stamp_ns,
            sequence=self._seq_camera,
        )

    # ── CameraInfo ─────────────────────────────────────────────────

    def write_camera_info(self, stamp_ns: int) -> None:
        """Write /camera/camera_info as sensor_msgs/CameraInfo.

        Locked contract: plumb_bob, zero distortion, K from intrinsics.
        """
        fn = self._fn
        intr = self._intrinsics

        # CameraInfo uses flat row-major matrices (K, R, P) in ROS messages.
        K = [
            float(intr.fx), 0.0, float(intr.cx),
            0.0, float(intr.fy), float(intr.cy),
            0.0, 0.0, 1.0,
        ]
        R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        P = [
            float(intr.fx), 0.0, float(intr.cx), 0.0,
            0.0, float(intr.fy), float(intr.cy), 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        msg = {
            "header": {
                "stamp": _ns_to_stamp(stamp_ns),
                "frame_id": fn.camera_optical,
            },
            "height": intr.height,
            "width": intr.width,
            "distortion_model": "plumb_bob",
            "d": [0.0, 0.0, 0.0, 0.0, 0.0],
            "k": K,
            "r": R,
            "p": P,
            "binning_x": 0,
            "binning_y": 0,
            "roi": {
                "x_offset": 0,
                "y_offset": 0,
                "height": 0,
                "width": 0,
                "do_rectify": False,
            },
        }

        self._writer.write_message(
            topic="/camera/camera_info",
            schema=self._schemas["camera_info"],
            message=msg,
            log_time=stamp_ns,
            publish_time=stamp_ns,
            sequence=self._seq_camera,
        )
        self._seq_camera += 1
