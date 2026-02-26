"""Integer-tick timeline generation — avoids float drift."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Flag, auto

import numpy as np


class EventType(Flag):
    LIDAR = auto()
    CAMERA = auto()
    TF = auto()


@dataclass
class TimelineEvent:
    tick_ns: int         # integer nanoseconds (canonical key)
    t_s: float           # float seconds (for trajectory evaluation)
    events: EventType    # which sensors fire at this tick


def make_timestamps(duration_s: float, rate_hz: float) -> np.ndarray:
    """Generate timestamps using integer ticks to avoid float accumulation.

    Returns float64 array of seconds: [0.0, 1/rate, 2/rate, ...].
    Last timestamp is strictly < duration_s.
    """
    # Floor via int(...) keeps the last timestamp strictly before duration_s.
    n_frames = int(duration_s * rate_hz)
    return np.arange(n_frames, dtype=np.float64) / rate_hz


def build_timeline(duration_s: float,
                   lidar_hz: float,
                   camera_hz: float,
                   tf_hz: float) -> list[TimelineEvent]:
    """Build a merged, sorted timeline with event tags.

    Uses integer nanoseconds as canonical keys to avoid float-equality bugs.
    """
    def to_ns(t_s: np.ndarray) -> np.ndarray:
        # Round once to integer ticks so later merges do not depend on float
        # equality between independently generated sensor timestamp arrays.
        return np.round(t_s * 1e9).astype(np.int64)

    t_lidar = make_timestamps(duration_s, lidar_hz)
    t_camera = make_timestamps(duration_s, camera_hz)
    t_tf = make_timestamps(duration_s, tf_hz)

    # Build a map: timestamp tick -> sensors that should fire at that tick.
    tick_map: dict[int, EventType] = {}

    for ns in to_ns(t_lidar):
        tick_map[int(ns)] = tick_map.get(int(ns), EventType(0)) | EventType.LIDAR
    for ns in to_ns(t_camera):
        tick_map[int(ns)] = tick_map.get(int(ns), EventType(0)) | EventType.CAMERA
    for ns in to_ns(t_tf):
        tick_map[int(ns)] = tick_map.get(int(ns), EventType(0)) | EventType.TF

    # Convert the map to a sorted event list in time order.
    events = []
    for ns in sorted(tick_map):
        events.append(TimelineEvent(
            tick_ns=ns,
            t_s=ns / 1e9,
            events=tick_map[ns],
        ))

    return events
