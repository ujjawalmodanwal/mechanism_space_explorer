"""
Nanoreactor simulation trajectory analysis and event detection routines.
"""

from nrnet.nanoreactor.events import (
    Event,
    StateRun,
    find_events,
    merge_adjacent_runs,
    run_length,
    stable_runs,
)

__all__ = [
    "Event",
    "StateRun",
    "find_events",
    "merge_adjacent_runs",
    "run_length",
    "stable_runs",
]
