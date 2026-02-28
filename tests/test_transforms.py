"""Tests for sim.transforms — SE3 utilities."""

import numpy as np
import pytest

from sim.transforms import (
    invert_se3,
    is_valid_se3,
    make_T_camera_link_optical,
    pose_to_matrix,
    transform_points,
)


def test_pose_to_matrix_identity():
    T = pose_to_matrix(0, 0, 0, 0, 0, 0)
    np.testing.assert_allclose(T, np.eye(4), atol=1e-12)


def test_pose_to_matrix_translation_only():
    T = pose_to_matrix(1.0, 2.0, 3.0, 0, 0, 0)
    assert T[0, 3] == pytest.approx(1.0)
    assert T[1, 3] == pytest.approx(2.0)
    assert T[2, 3] == pytest.approx(3.0)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)


def test_pose_to_matrix_yaw_90():
    T = pose_to_matrix(0, 0, 0, 0, 0, np.pi / 2)
    # 90° yaw: X-forward becomes Y-forward
    R = T[:3, :3]
    x_axis = R @ np.array([1, 0, 0])
    np.testing.assert_allclose(x_axis, [0, 1, 0], atol=1e-12)


def test_pose_to_matrix_produces_valid_se3():
    T = pose_to_matrix(5.0, -3.0, 1.8, 0.1, -0.05, 1.2)
    assert is_valid_se3(T)


def test_invert_se3_roundtrip():
    T = pose_to_matrix(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
    T_inv = invert_se3(T)
    np.testing.assert_allclose(T @ T_inv, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(T_inv @ T, np.eye(4), atol=1e-12)


def test_transform_points():
    T = pose_to_matrix(10, 0, 0, 0, 0, 0)  # pure 10m X translation
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    result = transform_points(T, pts)
    expected = pts + np.array([10, 0, 0])
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_transform_points_rotation():
    T = pose_to_matrix(0, 0, 0, 0, 0, np.pi / 2)  # 90° yaw
    pts = np.array([[1, 0, 0]], dtype=np.float64)
    result = transform_points(T, pts)
    np.testing.assert_allclose(result, [[0, 1, 0]], atol=1e-12)


def test_is_valid_se3_rejects_bad_matrix():
    T = np.eye(4)
    T[:3, :3] *= 2.0  # scaled, not orthogonal
    assert not is_valid_se3(T)


def test_is_valid_se3_rejects_wrong_shape():
    assert not is_valid_se3(np.eye(3))


def test_make_T_camera_link_optical_is_valid_se3():
    T = make_T_camera_link_optical()
    assert is_valid_se3(T)
    # Pure rotation, no translation
    np.testing.assert_allclose(T[:3, 3], [0, 0, 0], atol=1e-12)