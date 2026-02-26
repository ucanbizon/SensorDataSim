"""SE3 transform utilities.  T_A_B converts points FROM B INTO A."""

from __future__ import annotations

import numpy as np


def pose_to_matrix(x: float, y: float, z: float,
                   roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a 4×4 SE3 matrix from position and Euler angles (radians).

    Rotation order: Rz(yaw) @ Ry(pitch) @ Rx(roll)  (extrinsic XYZ / intrinsic ZYX).
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [-sp,      cp * sr,                 cp * cr],
    ], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T



def invert_se3(T: np.ndarray) -> np.ndarray:
    """Invert a 4×4 SE3 matrix: T_B_A = invert_se3(T_A_B)."""
    R = T[:3, :3]
    t = T[:3, 3]
    # SE3 inverse is cheap: invert rotation with transpose, then translate.
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv



def transform_points(T_A_B: np.ndarray, points_B: np.ndarray) -> np.ndarray:
    """Transform (N, 3) points from frame B to frame A using T_A_B.

    p_A = R @ p_B + t
    """
    R = T_A_B[:3, :3]
    t = T_A_B[:3, 3]
    # Batched affine transform without building homogeneous coordinates.
    return (R @ points_B.T).T + t


def is_valid_se3(T: np.ndarray, atol: float = 1e-8) -> bool:
    """Check if T is a valid 4×4 SE3 matrix."""
    if T.shape != (4, 4):
        return False
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=atol):
        return False
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=atol):
        return False
    if abs(np.linalg.det(R) - 1.0) > atol:
        return False
    return True


def make_T_camera_link_optical() -> np.ndarray:
    """T_camera_link_optical: optical (X=right,Y=down,Z=fwd) → body (X=fwd,Y=left,Z=up)."""
    T = np.eye(4, dtype=np.float64)
    # Fixed REP-103 camera_link -> optical rotation used in TF and rendering.
    T[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    return T
