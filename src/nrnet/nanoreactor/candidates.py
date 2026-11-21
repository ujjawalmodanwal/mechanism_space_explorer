"""
Freeze step and candidate extraction for nanoreactor reaction networks.

Extracts, groups, relaxes, and serializes candidate reaction events and their
associated molecular species into reproducible frozen candidate sets.
"""

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import CalculationFailed

from nrnet.chem import (
    connectivity_graph,
    fragments,
    mapped_reaction_smiles,
    multiplicity,
    radical_count,
    reacting_atoms,
    species_key,
    species_labels,
    state_key,
)
from nrnet.engines import EngineSettings, attach, load_engine_settings
from nrnet.nanoreactor.events import Event, find_events
from nrnet.optimize import RelaxResult, relax

__all__ = [
    "FREEZE_FMAX_EV_PER_ANG",
    "FREEZE_MAX_STEPS",
    "PAIRS_PER_CANDIDATE",
    "Candidate",
    "CandidateSet",
    "EndpointPair",
    "Species",
    "build_candidates_from_group",
    "compile_candidate_stats",
    "extract_candidate_set",
]

FREEZE_FMAX_EV_PER_ANG: float = 0.05
FREEZE_MAX_STEPS: int = 500
PAIRS_PER_CANDIDATE: int = 3

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Species:
    """
    Chemical species extracted and relaxed from candidate endpoints.

    :param key: Canonical species key in formula-hash format.
    :type key: str
    :param atoms: Optimized 3D atomic structure without calculator.
    :type atoms: Atoms
    :param smiles: Canonical SMILES identifier or None.
    :type smiles: str or None
    :param inchikey: Standard InChIKey identifier or None.
    :type inchikey: str or None
    :param multiplicity: Spin multiplicity (2S + 1).
    :type multiplicity: int
    """

    key: str
    atoms: Atoms = field(compare=False)
    smiles: str | None
    inchikey: str | None
    multiplicity: int


@dataclass(frozen=True, slots=True)
class EndpointPair:
    """
    Pair of relaxed endpoint geometries for an individual reaction event.

    :param replicate: Identifier of trajectory replicate where event occurred.
    :type replicate: int or tuple[str, int]
    :param time_fs: Timestamp of event onset in femtoseconds.
    :type time_fs: float
    :param atom_ids: Indices of reacting atoms mapped into the system frame.
    :type atom_ids: tuple[int, ...]
    :param before: Relaxed initial atomic geometry of reacting atoms.
    :type before: Atoms
    :param after: Relaxed final atomic geometry of reacting atoms.
    :type after: Atoms
    """

    replicate: int | tuple[str, int]
    time_fs: float
    atom_ids: tuple[int, ...]
    before: Atoms = field(compare=False)
    after: Atoms = field(compare=False)


@dataclass(frozen=True, slots=True)
class Candidate:
    """
    Candidate reaction joining two system states with validated endpoint pairs.

    :param reaction_id: Unique candidate reaction identifier (e.g. 'R001').
    :type reaction_id: str
    :param state_a: Canonical chemical state key tuple for initial state.
    :type state_a: tuple[str, ...]
    :param state_b: Canonical chemical state key tuple for final state.
    :type state_b: tuple[str, ...]
    :param reactant_keys: Expected species keys of reacting fragments on side A.
    :type reactant_keys: tuple[str, ...]
    :param product_keys: Expected species keys of reacting fragments on side B.
    :type product_keys: tuple[str, ...]
    :param pairs: Validated endpoint pairs for this candidate reaction.
    :type pairs: tuple[EndpointPair, ...]
    :param mapped_smiles: Atom-mapped reaction SMILES from first accepted pair.
    :type mapped_smiles: str
    :param n_events: Total count of raw events grouped into this candidate.
    :type n_events: int
    :param first_seen_fs: Timestamp in femtoseconds of earliest observed event.
    :type first_seen_fs: float
    :param source: Origin of candidate reaction ('md' or 'literature').
    :type source: str
    :param literature_name: Hand-built literature identifier name if matched.
    :type literature_name: str or None
    """

    reaction_id: str
    state_a: tuple[str, ...]
    state_b: tuple[str, ...]
    reactant_keys: tuple[str, ...]
    product_keys: tuple[str, ...]
    pairs: tuple[EndpointPair, ...]
    mapped_smiles: str
    n_events: int
    first_seen_fs: float
    source: str = "md"
    literature_name: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """
    Collection of frozen candidate reactions and associated molecular species.

    :param reactant_state: Initial state key tuple of the unreacted system.
    :type reactant_state: tuple[str, ...]
    :param species: Map of canonical species key to Species dataclass.
    :type species: dict[str, Species]
    :param candidates: Tuple of validated candidate reactions.
    :type candidates: tuple[Candidate, ...]
    :param stats: Summary counts of events, groups, species, and rejections.
    :type stats: dict[str, int]
    """

    reactant_state: tuple[str, ...]
    species: dict[str, Species]
    candidates: tuple[Candidate, ...]
    stats: dict[str, int] = field(default_factory=dict)


def collect_events(
    trajectories: Sequence[Sequence[Atoms]],
    min_frames: int,
    dump_fs: float,
    replicate_ids: Sequence[int | tuple[str, int]] | None = None,
) -> list[Event]:
    """
    Detect reaction events across all trajectory replicates.

    :param trajectories: Sequence of trajectory frame sequences.
    :type trajectories: Sequence[Sequence[Atoms]]
    :param min_frames: Minimum required duration in frames for stable runs.
    :type min_frames: int
    :param dump_fs: Trajectory frame dump interval in femtoseconds.
    :type dump_fs: float
    :param replicate_ids: Optional custom replicate identity per trajectory.
    :type replicate_ids: Sequence[int or tuple[str, int]] or None
    :return: Flat list of extracted reaction events.
    :rtype: list[Event]
    """
    events: list[Event] = []
    for idx, frames in enumerate(trajectories):
        rep = replicate_ids[idx] if replicate_ids is not None else idx
        found = find_events(
            frames, rep, min_frames, dump_fs, settle_frames=min_frames
        )
        events.extend(found)
    return events


def orient_event(event: Event, state_a: tuple[str, ...]) -> Event:
    """
    Orient an event towards canonical state_a -> state_b direction.

    :param event: Trajectory reaction event to orient.
    :type event: Event
    :param state_a: Canonical initial state key tuple.
    :type state_a: tuple[str, ...]
    :return: Oriented event with swapped frames if originally reversed.
    :rtype: Event
    """
    if event.state_before == state_a:
        return event
    return Event(
        replicate=event.replicate,
        time_fs=event.time_fs,
        frame_before=event.frame_after,
        frame_after=event.frame_before,
        state_before=event.state_after,
        state_after=event.state_before,
    )


def orient_group(group: Sequence[Event]) -> list[Event]:
    """
    Sort and orient an event group according to its earliest observed event.

    :param group: Sequence of events sharing the same unordered state pair.
    :type group: Sequence[Event]
    :return: List of oriented events sorted chronologically.
    :rtype: list[Event]
    """
    sorted_group = sorted(group, key=lambda ev: (ev.replicate, ev.time_fs))
    state_a = sorted_group[0].state_before
    return [orient_event(ev, state_a) for ev in sorted_group]


def group_by_state_pair(events: Sequence[Event]) -> list[list[Event]]:
    """
    Group events by unordered state pair and sort groups by earliest event.

    :param events: Sequence of raw reaction events.
    :type events: Sequence[Event]
    :return: Chronologically sorted list of oriented event groups.
    :rtype: list[list[Event]]
    """
    raw_groups: dict[tuple[tuple[str, ...], ...], list[Event]] = {}
    for ev in events:
        if ev.state_before == ev.state_after:
            LOGGER.info("Rejected identical state in rep %s", ev.replicate)
            continue
        key = tuple(sorted([ev.state_before, ev.state_after]))
        raw_groups.setdefault(key, []).append(ev)
    oriented_groups = [orient_group(g) for g in raw_groups.values()]
    oriented_groups.sort(key=lambda g: (g[0].replicate, g[0].time_fs))
    return oriented_groups


def reacting_atom_ids(before: Atoms, after: Atoms) -> list[int]:
    """
    Identify indices of reacting atoms grown to whole molecules.

    :param before: Initial atomic structure frame.
    :type before: Atoms
    :param after: Final atomic structure frame.
    :type after: Atoms
    :return: Sorted list of reacting atom indices.
    :rtype: list[int]
    """
    graph_before = connectivity_graph(before)
    graph_after = connectivity_graph(after)
    return reacting_atoms(graph_before, graph_after)


def expected_keys_for_frame(
    frame: Atoms,
    atom_ids: Sequence[int],
) -> tuple[str, ...]:
    """
    Compute expected species keys for fragments containing reacting atoms.

    :param frame: Raw atomic structure frame.
    :type frame: Atoms
    :param atom_ids: Indices of reacting atoms.
    :type atom_ids: Sequence[int]
    :return: Sorted tuple of fragment species keys.
    :rtype: tuple[str, ...]
    """
    graph = connectivity_graph(frame)
    reacting_set = set(atom_ids)
    matching_frags = [
        frag for frag in fragments(graph) if reacting_set.intersection(frag)
    ]
    keys = [species_key(graph, frag) for frag in matching_frags]
    return tuple(sorted(keys))


def relax_subset(
    frame: Atoms,
    atom_ids: Sequence[int],
    settings: EngineSettings,
) -> RelaxResult:
    """
    Cut reacting atom subset from a frame and relax with the engine.

    :param frame: Source atomic structure.
    :type frame: Atoms
    :param atom_ids: Indices of atoms to cut and relax.
    :type atom_ids: Sequence[int]
    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :return: Result of geometry optimization.
    :rtype: RelaxResult
    """
    subset = frame[list(atom_ids)]
    attached = attach(subset, settings)
    try:
        return relax(
            attached,
            fmax_ev_per_ang=FREEZE_FMAX_EV_PER_ANG,
            max_steps=FREEZE_MAX_STEPS,
        )
    except CalculationFailed:
        # SCF failures on hot, distorted fragments count as unconverged.
        return RelaxResult(atoms=subset, converged=False, steps=0)


def validate_pair_states(
    res_b: RelaxResult,
    res_a: RelaxResult,
    exp_b: tuple[str, ...],
    exp_a: tuple[str, ...],
) -> str | None:
    """
    Validate convergence and state keys for a relaxed endpoint pair.

    :param res_b: Relaxation result for before endpoint.
    :type res_b: RelaxResult
    :param res_a: Relaxation result for after endpoint.
    :type res_a: RelaxResult
    :param exp_b: Expected species keys for before endpoint.
    :type exp_b: tuple[str, ...]
    :param exp_a: Expected species keys for after endpoint.
    :type exp_a: tuple[str, ...]
    :return: Rejection reason string if invalid, None if accepted.
    :rtype: str or None
    """
    if not (res_b.converged and res_a.converged):
        return "unconverged"
    key_b, key_a = state_key(res_b.atoms), state_key(res_a.atoms)
    if key_b != exp_b or key_a != exp_a:
        return "state_mismatch"
    if key_b == key_a:
        return "identical_states"
    if state_key(res_b.atoms, stereo=True) == state_key(
        res_a.atoms, stereo=True
    ):
        return "identical_states"
    return None


def make_endpoint_pair(
    event: Event,
    atom_ids: Sequence[int],
    res_b: RelaxResult,
    res_a: RelaxResult,
) -> EndpointPair:
    """
    Construct an EndpointPair dataclass instance from relaxed endpoints.

    :param event: Source reaction event.
    :type event: Event
    :param atom_ids: Participating reacting atom indices.
    :type atom_ids: Sequence[int]
    :param res_b: Relaxed initial endpoint structure result.
    :type res_b: RelaxResult
    :param res_a: Relaxed final endpoint structure result.
    :type res_a: RelaxResult
    :return: Assembled endpoint pair instance.
    :rtype: EndpointPair
    """
    return EndpointPair(
        replicate=event.replicate,
        time_fs=event.time_fs,
        atom_ids=tuple(atom_ids),
        before=res_b.atoms,
        after=res_a.atoms,
    )


def build_endpoint_pair(
    event: Event,
    settings: EngineSettings,
) -> tuple[EndpointPair | None, str | None]:
    """
    Extract, relax, and validate an endpoint pair from an event.

    :param event: Candidate reaction event.
    :type event: Event
    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :return: Tuple of validated pair (or None) and rejection reason (or None).
    :rtype: tuple[EndpointPair or None, str or None]
    """
    atom_ids = reacting_atom_ids(event.frame_before, event.frame_after)
    if not atom_ids:
        return None, "no_reacting_atoms"
    frames = (event.frame_before, event.frame_after)
    if max(radical_count(frame, atom_ids) for frame in frames) > 1:
        return None, "multiple_radicals"
    exp_b = expected_keys_for_frame(event.frame_before, atom_ids)
    exp_a = expected_keys_for_frame(event.frame_after, atom_ids)
    res_b = relax_subset(event.frame_before, atom_ids, settings)
    res_a = relax_subset(event.frame_after, atom_ids, settings)
    reason = validate_pair_states(res_b, res_a, exp_b, exp_a)
    if reason is not None:
        return None, reason
    return make_endpoint_pair(event, atom_ids, res_b, res_a), None


def record_rejected_pair(
    reason: str | None,
    rejections: Counter[str],
) -> None:
    """
    Log and tally a rejected endpoint pair reason in counter.

    :param reason: Rejection reason string if rejected.
    :type reason: str or None
    :param rejections: Rejection counter to update.
    :type rejections: Counter[str]
    :return: None.
    :rtype: None
    """
    if reason is not None:
        rejections[reason] += 1
        LOGGER.info("Rejected pair: %s", reason)


def record_accepted_pair(
    event: Event,
    pair: EndpointPair,
    pairs_by_stereo: dict[
        tuple[tuple[str, ...], tuple[str, ...]], list[EndpointPair]
    ],
    first_events: dict[tuple[tuple[str, ...], tuple[str, ...]], Event],
    counts: Counter[tuple[tuple[str, ...], tuple[str, ...]]],
) -> None:
    """
    Tally an accepted pair under its stereo-aware endpoint state pair.

    :param event: Reaction event producing the pair.
    :type event: Event
    :param pair: Relaxed endpoint pair.
    :type pair: EndpointPair
    :param pairs_by_stereo: Map of stereo state pair to accepted pairs.
    :type pairs_by_stereo: dict[tuple, list[EndpointPair]]
    :param first_events: Earliest event for each stereo state pair.
    :type first_events: dict[tuple, Event]
    :param counts: Total count of raw events producing each stereo pair.
    :type counts: Counter[tuple]
    :return: None.
    :rtype: None
    """
    stereo_pair = (
        state_key(pair.before, stereo=True),
        state_key(pair.after, stereo=True),
    )
    counts[stereo_pair] += 1
    if stereo_pair not in first_events:
        first_events[stereo_pair] = event
        pairs_by_stereo[stereo_pair] = []
    if len(pairs_by_stereo[stereo_pair]) < PAIRS_PER_CANDIDATE:
        pairs_by_stereo[stereo_pair].append(pair)


def collect_stereo_pairs(
    group: Sequence[Event],
    settings: EngineSettings,
    rejections: Counter[str],
) -> tuple[
    dict[tuple[tuple[str, ...], tuple[str, ...]], list[EndpointPair]],
    dict[tuple[tuple[str, ...], tuple[str, ...]], Event],
    Counter[tuple[tuple[str, ...], tuple[str, ...]]],
]:
    """
    Relax event pairs and group accepted pairs by stereo-aware endpoint states.

    :param group: Oriented events sharing an unordered state pair.
    :type group: Sequence[Event]
    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :param rejections: Mutable tally mapping rejection reasons to counts.
    :type rejections: Counter[str]
    :return: Triplet of pairs by stereo key, first events, and event counts.
    :rtype: tuple[dict, dict, Counter]
    """
    pairs_by_stereo: dict[
        tuple[tuple[str, ...], tuple[str, ...]], list[EndpointPair]
    ] = {}
    first_events: dict[tuple[tuple[str, ...], tuple[str, ...]], Event] = {}
    counts: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
    for event in group:
        pair, reason = build_endpoint_pair(event, settings)
        if pair is not None:
            record_accepted_pair(
                event, pair, pairs_by_stereo, first_events, counts
            )
        record_rejected_pair(reason, rejections)
    return pairs_by_stereo, first_events, counts


def collect_candidate_pairs(
    events: Sequence[Event],
    settings: EngineSettings,
    rejections: Counter[str],
) -> list[EndpointPair]:
    """
    Collect accepted endpoint pairs for an event sequence across stereo types.

    :param events: Oriented event sequence.
    :type events: Sequence[Event]
    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :param rejections: Mutable tally mapping rejection reasons to counts.
    :type rejections: Counter[str]
    :return: List of accepted relaxed endpoint pairs.
    :rtype: list[EndpointPair]
    """
    pairs_map, _, _ = collect_stereo_pairs(events, settings, rejections)
    collected: list[EndpointPair] = []
    for pairs in pairs_map.values():
        collected.extend(pairs)
    return collected


def pair_mapped_smiles(pair: EndpointPair) -> str:
    """
    Compute atom-mapped reaction SMILES from relaxed endpoint geometries.

    :param pair: Relaxed endpoint pair.
    :type pair: EndpointPair
    :return: Atom-mapped reaction SMILES with 1-based trajectory atom indices.
    :rtype: str
    """
    max_idx = max(pair.atom_ids) + 1
    full_b = Atoms(["H"] * max_idx, positions=np.zeros((max_idx, 3)))
    full_a = Atoms(["H"] * max_idx, positions=np.zeros((max_idx, 3)))
    for pos_idx, orig_idx in enumerate(pair.atom_ids):
        full_b.symbols[orig_idx] = pair.before.symbols[pos_idx]
        full_b.positions[orig_idx] = pair.before.positions[pos_idx]
        full_a.symbols[orig_idx] = pair.after.symbols[pos_idx]
        full_a.positions[orig_idx] = pair.after.positions[pos_idx]
    return mapped_reaction_smiles(full_b, full_a, atom_ids=pair.atom_ids)


def candidate_endpoint_keys(
    pair: EndpointPair,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Extract reactant and product species keys for candidate endpoint pair.

    :param pair: Relaxed endpoint pair.
    :type pair: EndpointPair
    :return: Pair of stereo-aware reactant keys and product keys.
    :rtype: tuple[tuple[str, ...], tuple[str, ...]]
    """
    return (
        state_key(pair.before, stereo=True),
        state_key(pair.after, stereo=True),
    )


def candidate_reaction_meta(
    event: Event,
    pair: EndpointPair,
) -> dict[str, Any]:
    """
    Extract state keys and reaction smiles for candidate endpoints.

    :param event: Representative reaction event.
    :type event: Event
    :param pair: First validated endpoint pair.
    :type pair: EndpointPair
    :return: Dictionary containing states, species keys, and mapped SMILES.
    :rtype: dict[str, Any]
    """
    r_keys, p_keys = candidate_endpoint_keys(pair)
    return {
        "state_a": event.state_before,
        "state_b": event.state_after,
        "reactant_keys": r_keys,
        "product_keys": p_keys,
        "mapped_smiles": pair_mapped_smiles(pair),
    }


def assemble_candidate(
    reaction_id: str,
    event: Event,
    pairs: Sequence[EndpointPair],
    n_events: int,
) -> Candidate:
    """
    Construct a Candidate dataclass instance from validated pairs.

    :param reaction_id: Formatted reaction identifier (e.g. 'R001').
    :type reaction_id: str
    :param event: Representative earliest event for this candidate.
    :type event: Event
    :param pairs: Sequence of validated endpoint pairs.
    :type pairs: Sequence[EndpointPair]
    :param n_events: Total event count in candidate group.
    :type n_events: int
    :return: Assembled Candidate dataclass.
    :rtype: Candidate
    """
    meta = candidate_reaction_meta(event, pairs[0])
    return Candidate(
        reaction_id=reaction_id,
        pairs=tuple(pairs),
        n_events=n_events,
        first_seen_fs=event.time_fs,
        **meta,
    )


def build_candidates_from_group(
    group: Sequence[Event],
    settings: EngineSettings,
    rejections: Counter[str],
    candidate_start_idx: int,
) -> list[Candidate]:
    """
    Build Candidate reactions for each stereo-aware pair from an event group.

    :param group: Oriented events for a single unordered state pair.
    :type group: Sequence[Event]
    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :param rejections: Mutable tally mapping rejection reasons to counts.
    :type rejections: Counter[str]
    :param candidate_start_idx: Starting 1-based candidate index.
    :type candidate_start_idx: int
    :return: List of validated Candidate instances.
    :rtype: list[Candidate]
    """
    pairs_map, first_events, counts = collect_stereo_pairs(
        group, settings, rejections
    )
    sorted_pairs = sorted(
        pairs_map.keys(),
        key=lambda sp: (first_events[sp].replicate, first_events[sp].time_fs),
    )
    candidates: list[Candidate] = []
    for offset, stereo_pair in enumerate(sorted_pairs):
        n_events = len(group) if len(sorted_pairs) == 1 else counts[stereo_pair]
        cand_id = f"R{candidate_start_idx + offset:03d}"
        candidates.append(
            assemble_candidate(
                reaction_id=cand_id,
                event=first_events[stereo_pair],
                pairs=pairs_map[stereo_pair],
                n_events=n_events,
            )
        )
    return candidates


def build_candidate(
    group: Sequence[Event],
    settings: EngineSettings,
    rejections: Counter[str],
    candidate_idx: int,
) -> Candidate | None:
    """
    Build first Candidate reaction from an oriented event group.

    :param group: Oriented events for a single unordered state pair.
    :type group: Sequence[Event]
    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :param rejections: Mutable tally mapping rejection reasons to counts.
    :type rejections: Counter[str]
    :param candidate_idx: 1-based candidate index for ID formatting.
    :type candidate_idx: int
    :return: First Candidate or None if no pairs were accepted.
    :rtype: Candidate or None
    """
    candidates = build_candidates_from_group(
        group, settings, rejections, candidate_idx
    )
    return candidates[0] if candidates else None


def process_candidates(
    groups: Sequence[Sequence[Event]],
    settings: EngineSettings,
    rejections: Counter[str] | None = None,
) -> list[Candidate]:
    """
    Process candidate event groups into validated Candidate instances.

    :param groups: Sequence of oriented event groups.
    :type groups: Sequence[Sequence[Event]]
    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :param rejections: Optional tally for rejected pair reasons.
    :type rejections: Counter[str] or None
    :return: List of accepted candidates in earliest-event order.
    :rtype: list[Candidate]
    """
    tally = Counter() if rejections is None else rejections
    candidates: list[Candidate] = []
    for group in groups:
        new_cands = build_candidates_from_group(
            group, settings, tally, len(candidates) + 1
        )
        candidates.extend(new_cands)
    return candidates


def make_species(key: str, atoms: Atoms) -> Species:
    """
    Construct a Species dataclass instance with perceived chemical labels.

    :param key: Canonical species key.
    :type key: str
    :param atoms: Optimized molecular structure.
    :type atoms: Atoms
    :return: Instantiated Species dataclass.
    :rtype: Species
    """
    labels = species_labels(atoms)
    return Species(
        key=key,
        atoms=atoms,
        smiles=labels.smiles,
        inchikey=labels.inchikey,
        multiplicity=multiplicity(atoms),
    )


def relax_species(
    fragment: Atoms,
    key: str,
    settings: EngineSettings,
) -> Species | None:
    """
    Relax isolated molecular fragment and verify species key preservation.

    :param fragment: Cut molecular fragment structure.
    :type fragment: Atoms
    :param key: Expected canonical species key.
    :type key: str
    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :return: Species dataclass if relaxation converged to same key, else None.
    :rtype: Species or None
    """
    attached = attach(fragment, settings)
    res = relax(attached, FREEZE_FMAX_EV_PER_ANG, FREEZE_MAX_STEPS)
    if not res.converged or state_key(res.atoms, stereo=True) != (key,):
        return None
    return make_species(key, res.atoms)


def find_fragment_in_atoms(atoms: Atoms, target_key: str) -> Atoms | None:
    """
    Find and cut a fragment matching the target species key from Atoms.

    :param atoms: Source atomic structure.
    :type atoms: Atoms
    :param target_key: Target species key.
    :type target_key: str
    :return: Cut fragment Atoms object or None if not present.
    :rtype: Atoms or None
    """
    graph = connectivity_graph(atoms)
    for frag in fragments(graph):
        if species_key(graph, frag, atoms=atoms, stereo=True) == target_key:
            return atoms[frag]
    return None


def find_fragment_in_candidates(
    candidates: Sequence[Candidate],
    target_key: str,
) -> Atoms | None:
    """
    Locate fragment matching target species key in candidate first pairs.

    :param candidates: Sequence of accepted Candidate reactions.
    :type candidates: Sequence[Candidate]
    :param target_key: Target species key.
    :type target_key: str
    :return: Cut fragment Atoms object or None if not found.
    :rtype: Atoms or None
    """
    for cand in candidates:
        pair = cand.pairs[0]
        found = find_fragment_in_atoms(pair.before, target_key)
        if found is None:
            found = find_fragment_in_atoms(pair.after, target_key)
        if found is not None:
            return found
    return None


def collect_species_keys(
    reactant: Atoms,
    candidates: Sequence[Candidate],
) -> list[str]:
    """
    Collect all unique species keys required by reactant and candidates.

    :param reactant: Initial reactant state structure.
    :type reactant: Atoms
    :param candidates: Sequence of candidate reactions.
    :type candidates: Sequence[Candidate]
    :return: Sorted list of unique canonical species keys.
    :rtype: list[str]
    """
    needed: set[str] = set(state_key(reactant, stereo=True))
    for cand in candidates:
        needed.update(cand.reactant_keys)
        needed.update(cand.product_keys)
    return sorted(needed)


def find_species_fragment(
    key: str,
    reactant: Atoms,
    candidates: Sequence[Candidate],
) -> Atoms | None:
    """
    Find molecular fragment matching species key in reactant or candidates.

    :param key: Canonical species key.
    :type key: str
    :param reactant: Reactant structure frame.
    :type reactant: Atoms
    :param candidates: Candidate reactions sequence.
    :type candidates: Sequence[Candidate]
    :return: Fragment Atoms object or None if not found.
    :rtype: Atoms or None
    """
    frag = find_fragment_in_atoms(reactant, key)
    if frag is None:
        frag = find_fragment_in_candidates(candidates, key)
    return frag


def build_species_dict(
    reactant: Atoms,
    candidates: Sequence[Candidate],
    settings: EngineSettings,
) -> dict[str, Species]:
    """
    Build relaxed Species map for reactant state and all kept candidates.

    :param reactant: Initial reactant state structure.
    :type reactant: Atoms
    :param candidates: Sequence of kept candidate reactions.
    :type candidates: Sequence[Candidate]
    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :return: Dictionary mapping canonical species keys to Species instances.
    :rtype: dict[str, Species]
    """
    species_map: dict[str, Species] = {}
    for key in collect_species_keys(reactant, candidates):
        frag = find_species_fragment(key, reactant, candidates)
        if frag is not None:
            spec = relax_species(frag, key, settings)
            if spec is not None:
                species_map[key] = spec
    return species_map


def resolve_engine_settings(
    settings: EngineSettings | str | Path,
) -> EngineSettings:
    """
    Resolve engine settings input to an EngineSettings dataclass.

    :param settings: EngineSettings dataclass or path to YAML config.
    :type settings: EngineSettings or str or Path
    :return: Resolved EngineSettings instance for xtb.
    :rtype: EngineSettings
    """
    if isinstance(settings, EngineSettings):
        return settings
    return load_engine_settings(settings, "xtb")


def compile_candidate_stats(
    raw_events: Sequence[Event],
    groups: Sequence[Sequence[Event]],
    candidates: Sequence[Candidate],
    species_map: Mapping[str, Species],
    rejections: Counter[str],
) -> dict[str, int]:
    """
    Compile summary statistics for candidate set extraction.

    :param raw_events: Sequence of detected raw events.
    :type raw_events: Sequence[Event]
    :param groups: Sequence of grouped reaction events.
    :type groups: Sequence[Sequence[Event]]
    :param candidates: Sequence of accepted candidate reactions.
    :type candidates: Sequence[Candidate]
    :param species_map: Mapping of species keys to relaxed Species.
    :type species_map: Mapping[str, Species]
    :param rejections: Tally of rejected endpoint pairs by reason.
    :type rejections: Counter[str]
    :return: Summary statistics mapping metric names to integer counts.
    :rtype: dict[str, int]
    """
    stats: dict[str, int] = {
        "events": len(raw_events),
        "event_groups": len(groups),
        "candidates": len(candidates),
        "species": len(species_map),
        "reactive_replicates": len({ev.replicate for ev in raw_events}),
    }
    for reason, count in sorted(rejections.items()):
        stats[f"rejected_{reason}"] = count
    return stats


def extract_candidate_set(
    reactant: Atoms,
    trajectories: Sequence[Sequence[Atoms]],
    min_frames: int,
    dump_fs: float,
    engine_settings: EngineSettings | str | Path,
    replicate_ids: Sequence[int | tuple[str, int]] | None = None,
) -> CandidateSet:
    """
    Extract validated candidates and species map from trajectories.

    :param reactant: Initial reactant configuration.
    :type reactant: Atoms
    :param trajectories: Trajectory frame sequences.
    :type trajectories: Sequence[Sequence[Atoms]]
    :param min_frames: Minimum lifetime duration in frames.
    :type min_frames: int
    :param dump_fs: Trajectory frame dump interval in femtoseconds.
    :type dump_fs: float
    :param engine_settings: Configuration settings for relaxation engine.
    :type engine_settings: EngineSettings or str or Path
    :param replicate_ids: Optional custom replicate identity per trajectory.
    :type replicate_ids: Sequence[int or tuple[str, int]] or None
    :return: Unsaved CandidateSet instance.
    :rtype: CandidateSet
    """
    settings = resolve_engine_settings(engine_settings)
    raw_events = collect_events(
        trajectories, min_frames, dump_fs, replicate_ids
    )
    groups = group_by_state_pair(raw_events)
    rejections: Counter[str] = Counter()
    candidates = process_candidates(groups, settings, rejections)
    species_map = build_species_dict(reactant, candidates, settings)
    stats = compile_candidate_stats(
        raw_events, groups, candidates, species_map, rejections
    )
    return CandidateSet(
        state_key(reactant, stereo=True), species_map, tuple(candidates), stats
    )
