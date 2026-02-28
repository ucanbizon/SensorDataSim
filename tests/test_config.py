"""Tests for sim.config — YAML loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from sim.config import SimConfig, load_config


# Use the real sim.yaml as the baseline for valid config tests.
CONFIG_PATH = Path(__file__).resolve().parents[1] / "sim.yaml"


def test_load_config_returns_simconfig():
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, SimConfig)


def test_load_config_timeline_fields():
    cfg = load_config(CONFIG_PATH)
    assert cfg.timeline.duration_s == 15.0
    assert cfg.timeline.lidar_rate_hz == 10.0
    assert cfg.timeline.tf_rate_hz == 50.0


def test_load_config_ego_waypoints():
    cfg = load_config(CONFIG_PATH)
    assert len(cfg.ego.waypoints) >= 2
    # Waypoints are stored as (Y, X) tuples
    for wp in cfg.ego.waypoints:
        assert len(wp) == 2


def test_load_config_sensor_mounts():
    cfg = load_config(CONFIG_PATH)
    assert "velodyne" in cfg.sensor_mounts
    assert "camera_link" in cfg.sensor_mounts


def test_load_config_camera_intrinsics():
    cfg = load_config(CONFIG_PATH)
    assert cfg.camera_intrinsics.width > 0
    assert cfg.camera_intrinsics.height > 0
    assert cfg.camera_intrinsics.fx > 0


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def _write_bad_config(base_path: Path, overrides: dict) -> Path:
    """Load the real config, apply overrides, write to a temp file."""
    with open(base_path) as f:
        raw = yaml.safe_load(f)
    for key_path, value in overrides.items():
        keys = key_path.split(".")
        d = raw
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(raw, tmp, sort_keys=False)
    tmp.close()
    return Path(tmp.name)


def test_validation_rejects_bad_lidar_range():
    path = _write_bad_config(CONFIG_PATH, {"lidar_render.min_range_m": -1.0})
    with pytest.raises(ValueError, match="lidar range"):
        load_config(path)


def test_validation_rejects_bad_camera_range():
    path = _write_bad_config(CONFIG_PATH, {"camera_render.min_range_m": 200.0})
    with pytest.raises(ValueError, match="camera range"):
        load_config(path)


def test_validation_rejects_bad_image_format():
    path = _write_bad_config(CONFIG_PATH, {"camera_render.image_format": "bmp"})
    with pytest.raises(ValueError, match="image format"):
        load_config(path)


def test_validation_rejects_bad_supersample():
    path = _write_bad_config(CONFIG_PATH, {"camera_render.supersample": 4})
    with pytest.raises(ValueError, match="Supersample"):
        load_config(path)