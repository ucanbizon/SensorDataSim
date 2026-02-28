"""Tests for sim.timeline — integer-tick timeline generation."""

import numpy as np
import pytest

from sim.timeline import EventType, TimelineEvent, build_timeline, make_timestamps


def test_make_timestamps_count():
    ts = make_timestamps(15.0, 10.0)
    assert len(ts) == 150


def test_make_timestamps_starts_at_zero():
    ts = make_timestamps(5.0, 10.0)
    assert ts[0] == pytest.approx(0.0)


def test_make_timestamps_stays_below_duration():
    ts = make_timestamps(15.0, 10.0)
    assert ts[-1] < 15.0


def test_make_timestamps_uniform_spacing():
    ts = make_timestamps(10.0, 20.0)
    diffs = np.diff(ts)
    np.testing.assert_allclose(diffs, 0.05, atol=1e-12)


def test_build_timeline_is_sorted():
    events = build_timeline(5.0, 10.0, 10.0, 50.0)
    ticks = [e.tick_ns for e in events]
    assert ticks == sorted(ticks)


def test_build_timeline_no_duplicates():
    events = build_timeline(5.0, 10.0, 10.0, 50.0)
    ticks = [e.tick_ns for e in events]
    assert len(ticks) == len(set(ticks))


def test_build_timeline_first_tick_has_all_events():
    events = build_timeline(5.0, 10.0, 10.0, 50.0)
    first = events[0]
    assert first.tick_ns == 0
    assert EventType.LIDAR in first.events
    assert EventType.CAMERA in first.events
    assert EventType.TF in first.events


def test_event_type_flags_combine():
    combined = EventType.LIDAR | EventType.CAMERA
    assert EventType.LIDAR in combined
    assert EventType.CAMERA in combined
    assert EventType.TF not in combined


def test_build_timeline_t_s_matches_tick_ns():
    events = build_timeline(2.0, 10.0, 10.0, 50.0)
    for e in events:
        assert e.t_s == pytest.approx(e.tick_ns / 1e9, abs=1e-12)