"""
Event detection and lifetime filtering for nanoreactor trajectories.

Extracts chemical state runs and detects discrete transition events between
consecutive stable molecular states along an ab initio molecular dynamics path.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

from ase import Atoms

from nrnet.chem import state_key

__all__ = [
    "Event",
    "StateRun",
    "endpoint_index",
    "find_events",
    "merge_adjacent_runs",
    "run_length",
    "stable_runs",
]


@dataclass(frozen=True, slots=True)
class StateRun:
    """
    Continuous run of an identical chemical state in a trajectory.

    :param state: Chemical state tuple representing molecular species present.
    :type state: tuple[str, ...]
    :param start: Starting frame index (inclusive).
    :type start: int
    :param stop: Ending frame index (exclusive).
    :type stop: int
    """

    state: tuple[str, ...]
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class Event:
    """
    Reaction event between two consecutive stable states in a trajectory.

    :param replicate: Identifier of MD replicate where event occurred.
    :type replicate: int or tuple[str, int]
    :param time_fs: Simulation timestamp of event onset in femtoseconds.
    :type time_fs: float
    :param frame_before: Final atomic structure before transition.
    :type frame_before: Atoms
    :param frame_after: Initial atomic structure after transition.
    :type frame_after: Atoms
    :param state_before: Chemical state tuple before transition.
    :type state_before: tuple[str, ...]
    :param state_after: Chemical state tuple after transition.
    :type state_after: tuple[str, ...]
    """

    replicate: int | tuple[str, int]
    time_fs: float
    frame_before: Atoms = field(compare=False)
    frame_after: Atoms = field(compare=False)
    state_before: tuple[str, ...]
    state_after: tuple[str, ...]


def run_length(states: Sequence[tuple[str, ...]]) -> list[StateRun]:
    """
    Compute plain run-length encoding for a sequence of chemical states.

    :param states: Sequence of chemical state tuples.
    :type states: Sequence[tuple[str, ...]]
    :return: List of continuous state runs with start and stop frame indices.
    :rtype: list[StateRun]
    """
    runs: list[StateRun] = []
    current_idx = 0
    for key, group in itertools.groupby(states):
        count = sum(1 for _ in group)
        runs.append(
            StateRun(state=key, start=current_idx, stop=current_idx + count)
        )
        current_idx += count
    return runs


def merge_adjacent_runs(runs: Sequence[StateRun]) -> list[StateRun]:
    """
    Merge adjacent state runs that share identical chemical states.

    :param runs: Sequence of state runs to merge.
    :type runs: Sequence[StateRun]
    :return: List of merged state runs where adjacent runs have distinct states.
    :rtype: list[StateRun]
    """
    merged: list[StateRun] = []
    for state, group in itertools.groupby(runs, key=lambda run: run.state):
        items = list(group)
        merged.append(
            StateRun(
                state=state,
                start=items[0].start,
                stop=items[-1].stop,
            )
        )
    return merged


def stable_runs(
    states: Sequence[tuple[str, ...]],
    min_frames: int,
) -> list[StateRun]:
    """
    Filter transient state runs and merge adjacent surviving runs.

    Runs shorter than min_frames are dropped, except the first run which is
    always kept as the known reactant start. Adjacent surviving runs that
    share the same state are merged.

    :param states: Sequence of chemical state tuples.
    :type states: Sequence[tuple[str, ...]]
    :param min_frames: Minimum required duration in frames for stability.
    :type min_frames: int
    :return: List of stable state runs.
    :rtype: list[StateRun]
    """
    raw_runs = run_length(states)
    filtered = [
        run
        for idx, run in enumerate(raw_runs)
        if idx == 0 or (run.stop - run.start) >= min_frames
    ]
    return merge_adjacent_runs(filtered)


def endpoint_index(
    states: Sequence[tuple[str, ...]],
    run: StateRun,
    before: bool,
    settle_frames: int,
) -> int:
    """
    Pick a run's endpoint frame settle_frames away from the crossing.

    A frame right at the crossing often sits on the barrier and relaxes
    to the other side, which the endpoint check rejects (18 of 21
    cpd_ma events at 0.01 Eh per atom, 2026-11-18). The frame chosen is
    the nearest one at or past the target, toward the crossing, that is
    in the run's own state.

    :param states: State key of every frame.
    :type states: Sequence[tuple[str, ...]]
    :param run: The stable run to take the endpoint from.
    :type run: StateRun
    :param before: True for the run before the crossing, False after.
    :type before: bool
    :param settle_frames: Frames to step away from the crossing.
    :type settle_frames: int
    :return: Frame index of the endpoint.
    :rtype: int
    """
    if before:
        target = max(run.start, run.stop - 1 - settle_frames)
        order = range(target, run.stop)
    else:
        target = min(run.stop - 1, run.start + settle_frames)
        order = range(target, run.start - 1, -1)
    return next(i for i in order if states[i] == run.state)


def find_events(
    frames: Sequence[Atoms],
    replicate: int | tuple[str, int],
    min_frames: int,
    dump_fs: float,
    scale: float = 1.2,
    settle_frames: int = 0,
) -> list[Event]:
    """
    Detect reaction events between consecutive stable trajectory runs.

    :param frames: Sequence of atomic structures along the MD trajectory.
    :type frames: Sequence[Atoms]
    :param replicate: Replicate identifier for the trajectory.
    :type replicate: int or tuple[str, int]
    :param min_frames: Minimum lifetime in frames to qualify as stable state.
    :type min_frames: int
    :param dump_fs: Trajectory dumping interval in femtoseconds.
    :type dump_fs: float
    :param scale: Multiplicative scale factor for covalent cutoffs.
    :type scale: float
    :param settle_frames: Frames between the crossing and each endpoint.
    :type settle_frames: int
    :return: List of reaction events extracted between consecutive stable runs.
    :rtype: list[Event]
    """
    states = [state_key(frame, scale=scale) for frame in frames]
    runs = stable_runs(states, min_frames=min_frames)
    return [
        Event(
            replicate=replicate,
            time_fs=float(later.start * dump_fs),
            frame_before=frames[
                endpoint_index(states, earlier, True, settle_frames)
            ],
            frame_after=frames[
                endpoint_index(states, later, False, settle_frames)
            ],
            state_before=earlier.state,
            state_after=later.state,
        )
        for earlier, later in itertools.pairwise(runs)
    ]
