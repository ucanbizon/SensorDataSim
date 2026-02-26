"""Create a side-by-side LiDAR + camera video from one MCAP file.

Left panel:
    Simulated VLP-16 point cloud rendered as a simple top-down BEV image.
    Points are colored by the PointCloud2 intensity field, so dynamic agents
    (forced to high intensity in sim/lidar.py) stand out.

Right panel:
    Camera /camera/image_raw frames (RGB8) from the same MCAP.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from mcap_ros2.reader import read_ros2_messages


# PointCloud2 layout written by sim/mcap_writer.py (point_step = 20).
_PC2_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("intensity", "<f4"),
        ("ring", "<u2"),
        ("_pad", "<u2"),
    ]
)


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------

def _put_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float,
              color: tuple[int, int, int], thickness: int = 1,
              shadow: bool = True) -> None:
    """Draw text, optionally with a small shadow for contrast."""
    if shadow:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _draw_panel_label(canvas_bgr: np.ndarray, x0: int, width: int,
                      title: str, subtitle: str) -> None:
    """Draw a two-line header box on one half of the video canvas."""
    pad = 10
    box_h = 58
    x1 = x0 + width
    cv2.rectangle(canvas_bgr, (x0 + pad, pad), (x1 - pad, pad + box_h),
                  (22, 22, 22), -1)
    cv2.rectangle(canvas_bgr, (x0 + pad, pad), (x1 - pad, pad + box_h),
                  (180, 180, 180), 1)
    _put_text(canvas_bgr, title, (x0 + pad + 10, pad + 22), 0.65,
              (255, 255, 255), 2)
    _put_text(canvas_bgr, subtitle, (x0 + pad + 10, pad + 46), 0.52,
              (200, 230, 255), 1)


def _iter_camera_frames(mcap_path: Path, topic: str = "/camera/image_raw"):
    """Yield (log_time_ns, rgb_uint8_image) from the camera topic."""
    for m in read_ros2_messages(str(mcap_path), topics=[topic]):
        msg = m.ros_msg
        if str(getattr(msg, "encoding", "")).lower() != "rgb8":
            raise ValueError(f"Unsupported camera encoding {msg.encoding!r}")
        h = int(msg.height)
        w = int(msg.width)
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        # Copy to decouple from the reader buffer.
        yield int(m.log_time_ns), img.copy()


def _iter_lidar_frames(mcap_path: Path, topic: str = "/velodyne_points"):
    """Yield (log_time_ns, structured array) from the LiDAR PointCloud2 topic."""
    for m in read_ros2_messages(str(mcap_path), topics=[topic]):
        msg = m.ros_msg
        n = int(msg.width) * int(msg.height)
        if int(msg.point_step) != 20:
            raise ValueError(f"Unexpected point_step={msg.point_step}; expected 20")
        if len(msg.data) < n * 20:
            raise ValueError("PointCloud2 data shorter than expected from width/height")
        pts = np.frombuffer(msg.data, dtype=_PC2_DTYPE, count=n).copy()
        yield int(m.log_time_ns), pts


def _render_lidar_bev_panel(
    pts: np.ndarray,
    panel_w: int,
    panel_h: int,
    *,
    # Use a symmetric forward/back range so the ego marker stays centered.
    x_min_m: float = -50.0,
    x_max_m: float = 50.0,
    y_min_m: float = -35.0,
    y_max_m: float = 35.0,
) -> np.ndarray:
    """Render a top-down BEV image from one LiDAR frame.

    Coordinates assume the Velodyne frame uses x=forward, y=left, z=up.
    """
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    # White/light background makes low-intensity static points easier to see.
    panel[:] = (248, 248, 248)

    # Layout: BEV plot + notes area on the right.
    margin = 18
    # Wider notes area so text is readable in the exported video.
    legend_w = 320
    plot_w = max(100, panel_w - legend_w - 3 * margin)
    plot_h = panel_h - 2 * margin
    plot_x0 = margin
    plot_y0 = margin
    plot_x1 = plot_x0 + plot_w
    plot_y1 = plot_y0 + plot_h

    # Plot background + grid.
    cv2.rectangle(panel, (plot_x0, plot_y0), (plot_x1, plot_y1), (255, 255, 255), -1)
    cv2.rectangle(panel, (plot_x0, plot_y0), (plot_x1, plot_y1), (120, 120, 120), 1)

    # Major grid every 10 m. Grid helps reading motion/spacing in the BEV.
    for x_m in range(int(np.ceil(x_min_m / 10) * 10), int(np.floor(x_max_m / 10) * 10) + 1, 10):
        v = int(round(plot_y1 - (x_m - x_min_m) / (x_max_m - x_min_m) * plot_h))
        if plot_y0 <= v <= plot_y1:
            cv2.line(panel, (plot_x0, v), (plot_x1, v), (220, 220, 220), 1, cv2.LINE_AA)
    for y_m in range(int(np.ceil(y_min_m / 10) * 10), int(np.floor(y_max_m / 10) * 10) + 1, 10):
        u = int(round(plot_x0 + (y_m - y_min_m) / (y_max_m - y_min_m) * plot_w))
        if plot_x0 <= u <= plot_x1:
            cv2.line(panel, (u, plot_y0), (u, plot_y1), (220, 220, 220), 1, cv2.LINE_AA)

    # Ego marker at (x=0, y=0).
    ego_u = int(round(plot_x0 + (0.0 - y_min_m) / (y_max_m - y_min_m) * plot_w))
    ego_v = int(round(plot_y1 - (0.0 - x_min_m) / (x_max_m - x_min_m) * plot_h))
    cv2.circle(panel, (ego_u, ego_v), 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.arrowedLine(panel, (ego_u, ego_v), (ego_u, max(plot_y0, ego_v - 28)),
                    (0, 220, 255), 2, cv2.LINE_AA, tipLength=0.35)
    _put_text(panel, "ego", (ego_u + 8, ego_v - 6), 0.45, (255, 255, 255), 1)

    if pts.size == 0:
        _put_text(panel, "No LiDAR points", (plot_x0 + 20, plot_y0 + 40), 0.7, (40, 40, 40), 2)
        return panel

    x = pts["x"].astype(np.float32, copy=False)
    y = pts["y"].astype(np.float32, copy=False)
    z = pts["z"].astype(np.float32, copy=False)
    intensity = pts["intensity"].astype(np.float32, copy=False)

    # Loose z gate removes occasional extreme outliers and keeps the BEV cleaner.
    m = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(intensity)
        & (x >= x_min_m) & (x <= x_max_m)
        & (y >= y_min_m) & (y <= y_max_m)
        & (z >= -4.0) & (z <= 4.0)
    )
    if not np.any(m):
        _put_text(panel, "No points in BEV view", (plot_x0 + 20, plot_y0 + 40), 0.7, (40, 40, 40), 2)
        return panel

    x = x[m]
    y = y[m]
    intensity = np.clip(intensity[m], 0.0, 255.0)

    # Convert metric coordinates to pixel coordinates.
    # Flip horizontally so "left of ego" in the world appears on the left side
    # of the image (matches the camera view direction convention better).
    u = np.rint(plot_x1 - (y - y_min_m) / (y_max_m - y_min_m) * plot_w).astype(np.int32)
    v = np.rint(plot_y1 - (x - x_min_m) / (x_max_m - x_min_m) * plot_h).astype(np.int32)
    inb = (u >= plot_x0) & (u <= plot_x1) & (v >= plot_y0) & (v <= plot_y1)
    if not np.any(inb):
        return panel

    u = u[inb]
    v = v[inb]
    intensity_u8 = np.rint(intensity[inb]).astype(np.uint8)
    # The simulator compresses static intensities into 0..40 and reserves 255
    # for agents. Stretch the static band for this *video visualization* only,
    # otherwise the colormap packs most static points into very dark tones.
    disp_intensity_u8 = intensity_u8.copy()
    static_mask = disp_intensity_u8 < 240
    if np.any(static_mask):
        static_vals = disp_intensity_u8[static_mask].astype(np.float32)
        static_vals = np.clip(static_vals, 0.0, 40.0)
        # Map 0..40 -> 45..220 for better visibility on a white background.
        disp_intensity_u8[static_mask] = np.rint(45.0 + (static_vals / 40.0) * 175.0).astype(np.uint8)

    # Turbo gives a strong bright high end, which makes agent points (255) obvious.
    colors = cv2.applyColorMap(disp_intensity_u8.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)

    # Draw lower-intensity points first, then brighter points on top.
    order = np.argsort(intensity_u8, kind="stable")
    u = u[order]
    v = v[order]
    colors = colors[order]
    intensity_u8 = intensity_u8[order]

    panel[v, u] = colors

    # Enlarge only the brightest points (dynamic agents at 255) so they remain visible.
    bright = intensity_u8 >= 240
    if np.any(bright):
        for uu, vv, cc in zip(u[bright], v[bright], colors[bright]):
            cv2.circle(panel, (int(uu), int(vv)), 2, tuple(int(c) for c in cc), -1, cv2.LINE_AA)

    # Small legend / notes.
    lx = plot_x1 + 16
    # Keep the notes compact and readable. Put them on a light card and use
    # thin dark text (no shadow), which is easier to read than bold outlined
    # text on a bright background.
    card_x0 = lx - 8
    card_y0 = plot_y0 + 4
    card_x1 = panel_w - margin
    card_y1 = plot_y0 + 290
    card = panel[card_y0:card_y1, card_x0:card_x1]
    # Semi-transparent white card so the plot remains visible behind it.
    panel[card_y0:card_y1, card_x0:card_x1] = (
        0.80 * np.full_like(card, 255) + 0.20 * card
    ).astype(np.uint8)
    cv2.rectangle(panel, (card_x0, card_y0), (card_x1, card_y1), (180, 180, 180), 1)

    y_text = plot_y0 + 22
    line_h = 24
    _put_text(panel, "BEV (top-down)", (lx, y_text), 0.60, (25, 25, 25), 1, shadow=False)
    y_text += line_h
    _put_text(panel, "ego centered", (lx, y_text), 0.50, (50, 50, 50), 1, shadow=False)
    y_text += line_h
    _put_text(panel, "x = forward, y = left", (lx, y_text), 0.50, (50, 50, 50), 1, shadow=False)
    y_text += line_h
    _put_text(panel, "left/right matches camera", (lx, y_text), 0.48, (50, 50, 50), 1, shadow=False)
    y_text += line_h + 6
    _put_text(panel, f"points: {pts.shape[0]:,}", (lx, y_text), 0.50, (50, 50, 50), 1, shadow=False)
    y_text += line_h
    _put_text(panel, "color: intensity (display stretch)", (lx, y_text), 0.46, (50, 50, 50), 1, shadow=False)
    y_text += line_h
    _put_text(panel, "static: 0..40 -> stretched", (lx, y_text), 0.46, (50, 50, 50), 1, shadow=False)
    y_text += line_h
    _put_text(panel, "dynamic agents: 255 (bright)", (lx, y_text), 0.48, (130, 70, 20), 1, shadow=False)
    y_text += line_h + 10
    _put_text(panel, "BEV range:", (lx, y_text), 0.50, (25, 25, 25), 1, shadow=False)
    y_text += line_h
    _put_text(panel, f"x {x_min_m:.0f}..{x_max_m:.0f} m", (lx, y_text), 0.48, (50, 50, 50), 1, shadow=False)
    y_text += line_h
    _put_text(panel, f"y {y_min_m:.0f}..{y_max_m:.0f} m", (lx, y_text), 0.48, (50, 50, 50), 1, shadow=False)

    return panel


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------

def make_lidar_camera_video(
    mcap_path: Path,
    out_mp4: Path,
    fps: float = 10.0,
) -> None:
    """Read one MCAP and write a side-by-side LiDAR+camera MP4.

    The function truncates to the shorter stream if LiDAR and camera frame
    counts differ.
    """
    cam_iter = _iter_camera_frames(mcap_path)
    lidar_iter = _iter_lidar_frames(mcap_path)

    try:
        cam_t0, cam_img0 = next(cam_iter)
        lid_t0, lid_pts0 = next(lidar_iter)
    except StopIteration:
        raise RuntimeError("MCAP missing /camera/image_raw or /velodyne_points frames")

    h, w, _ = cam_img0.shape
    header_h = 82
    footer_h = 34
    out_h = header_h + h + footer_h
    out_w = w * 2

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_mp4}")

    frame_idx = 0
    max_rel_dt_ms = 0.0

    def _write_pair(i: int, lid_ts: int, lid_pts: np.ndarray, cam_ts: int, cam_rgb: np.ndarray) -> None:
        """Render one combined output frame from aligned LiDAR + camera frames."""
        nonlocal max_rel_dt_ms
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:] = (12, 12, 14)

        lidar_panel = _render_lidar_bev_panel(lid_pts, w, h)
        cam_bgr = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)

        canvas[header_h:header_h + h, 0:w] = lidar_panel
        canvas[header_h:header_h + h, w:2 * w] = cam_bgr

        cv2.line(canvas, (w, 0), (w, out_h), (90, 90, 90), 1, cv2.LINE_AA)
        cv2.line(canvas, (0, header_h), (out_w, header_h), (90, 90, 90), 1, cv2.LINE_AA)
        cv2.line(canvas, (0, header_h + h), (out_w, header_h + h), (90, 90, 90), 1, cv2.LINE_AA)

        _draw_panel_label(
            canvas, 0, w,
            "LiDAR (simulated VLP-16)",
            "16 rings, PointCloud2, BEV top-down, color by intensity (agents bright)",
        )
        _draw_panel_label(
            canvas, w, w,
            "Camera (pinhole RGB)",
            f"{w}x{h} RGB8, Gaussian depth blend + supersample/downsample",
        )

        t_cam_s = (cam_ts - cam_t0) / 1e9
        t_lid_s = (lid_ts - lid_t0) / 1e9
        rel_dt_ms = abs((cam_ts - cam_t0) - (lid_ts - lid_t0)) / 1e6
        max_rel_dt_ms = max(max_rel_dt_ms, rel_dt_ms)

        footer = (
            f"frame {i:03d}   lidar t={t_lid_s:5.2f}s   camera t={t_cam_s:5.2f}s   "
            f"rel dt={rel_dt_ms:5.1f} ms   source={mcap_path.name}"
        )
        _put_text(canvas, footer, (12, header_h + h + 24), 0.58, (220, 220, 220), 1)

        writer.write(canvas)

    # Write the already-read first pair before entering the streaming loop.
    _write_pair(frame_idx, lid_t0, lid_pts0, cam_t0, cam_img0)
    frame_idx += 1

    while True:
        try:
            lid_ts, lid_pts = next(lidar_iter)
            has_lid = True
        except StopIteration:
            has_lid = False
            lid_ts, lid_pts = 0, None
        try:
            cam_ts, cam_img = next(cam_iter)
            has_cam = True
        except StopIteration:
            has_cam = False
            cam_ts, cam_img = 0, None

        if not has_lid or not has_cam:
            if has_lid != has_cam:
                print("Warning: LiDAR and camera frame counts differ; output truncated to shortest stream.")
            break

        assert lid_pts is not None and cam_img is not None
        _write_pair(frame_idx, lid_ts, lid_pts, cam_ts, cam_img)
        frame_idx += 1

    writer.release()
    print(f"Wrote {frame_idx} frames -> {out_mp4}")
    print(f"Max relative timestamp difference (camera vs lidar): {max_rel_dt_ms:.3f} ms")


def main() -> int:
    ap = argparse.ArgumentParser(description="Make side-by-side LiDAR+camera video from one MCAP")
    ap.add_argument("--mcap", type=Path, required=True, help="Input MCAP path")
    ap.add_argument("--output", type=Path, required=True, help="Output MP4 path")
    ap.add_argument("--fps", type=float, default=10.0, help="Output video FPS (default: 10)")
    args = ap.parse_args()

    make_lidar_camera_video(args.mcap, args.output, fps=args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
