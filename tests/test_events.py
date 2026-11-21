"""
Unit tests for nanoreactor trajectory event detection and lifetime filtering.

Verifies run-length encoding, stable run filtering and merging, event
extraction from synthetic state sequences, and event detection on embedded
molecular frames.
"""

import pytest
from ase import Atoms
from ase.build import molecule
from rdkit import Chem
from rdkit.Chem import AllChem

from nrnet.chem import state_key
from nrnet.nanoreactor.events import (
    Event,
    StateRun,
    find_events,
    run_length,
    stable_runs,
)


def mol_to_atoms(mol: Chem.Mol) -> Atoms:
    """
    Convert an RDKit Mol with 3D coordinates to an ASE Atoms object.

    :param mol: Embedded RDKit molecule.
    :type mol: Chem.Mol
    :return: ASE Atoms representation.
    :rtype: Atoms
    """
    conformer = mol.GetConformer()
    positions = conformer.GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=positions)


def build_molecule_from_smiles(
    smiles: str,
    random_seed: int = 42,
) -> Atoms:
    """
    Generate an ASE Atoms object from SMILES via 3D embedding.

    :param smiles: Chemical SMILES string.
    :type smiles: str
    :param random_seed: Seed for reproducible conformer generation.
    :type random_seed: int
    :return: 3D embedded ASE Atoms instance.
    :rtype: Atoms
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=random_seed)
    AllChem.UFFOptimizeMolecule(mol)
    return mol_to_atoms(mol)


def reorder_atoms_to_symbols(
    atoms: Atoms,
    target_symbols: list[str],
) -> Atoms:
    """
    Permute atoms to match a specified sequence of elemental symbols.

    :param atoms: Source atomic structure.
    :type atoms: Atoms
    :param target_symbols: Target elemental symbol sequence.
    :type target_symbols: list[str]
    :return: Reordered Atoms instance.
    :rtype: Atoms
    """
    cur_symbols = atoms.get_chemical_symbols()
    used_indices: set[int] = set()
    permutation: list[int] = []
    for target_symbol in target_symbols:
        for idx, cur_symbol in enumerate(cur_symbols):
            if cur_symbol == target_symbol and idx not in used_indices:
                used_indices.add(idx)
                permutation.append(idx)
                break
    return atoms[permutation]


def sequence_to_states(pattern: str) -> list[tuple[str, ...]]:
    """
    Convert a string of state characters into a list of state tuples.

    :param pattern: Sequence string where each character represents a state.
    :type pattern: str
    :return: List of single-element state tuples.
    :rtype: list[tuple[str, ...]]
    """
    return [(char,) for char in pattern]


DUMMY_ELEMENTS: dict[str, Atoms] = {
    "A": Atoms("H"),
    "B": Atoms("He"),
    "C": Atoms("Li"),
}

STATE_A: tuple[str, ...] = state_key(DUMMY_ELEMENTS["A"])
STATE_B: tuple[str, ...] = state_key(DUMMY_ELEMENTS["B"])


def make_dummy_frames(pattern: str) -> list[Atoms]:
    """
    Map a sequence string to distinct single-atom ASE structures.

    :param pattern: Sequence string where letters select distinct elements.
    :type pattern: str
    :return: List of ASE Atoms instances corresponding to each character.
    :rtype: list[Atoms]
    """
    return [DUMMY_ELEMENTS[char].copy() for char in pattern]


def expected_state_tuple(
    state: str | tuple[str, ...],
) -> tuple[str, ...]:
    """
    Resolve letter code or state key to standard chemical state tuple.

    :param state: Element code or chemical state tuple.
    :type state: str | tuple[str, ...]
    :return: Canonical chemical state key tuple.
    :rtype: tuple[str, ...]
    """
    if isinstance(state, str):
        return state_key(DUMMY_ELEMENTS[state])
    return state


def make_event(
    time_fs: float,
    symbols: tuple[str, str],
) -> Event:
    """
    Construct test Event with specified timestamp and dummy endpoint frames.

    :param time_fs: Event timestamp in femtoseconds.
    :type time_fs: float
    :param symbols: Pair of elemental symbols for before and after frames.
    :type symbols: tuple[str, str]
    :return: Configured Event instance.
    :rtype: Event
    """
    return Event(
        replicate=0,
        time_fs=time_fs,
        frame_before=Atoms(symbols[0]),
        frame_after=Atoms(symbols[1]),
        state_before=("A",),
        state_after=("B",),
    )


def build_hcho_nh3_frames() -> tuple[list[Atoms], Atoms, Atoms]:
    """
    Build 10-frame HCHO+NH3 to aminomethanol product trajectory.

    :return: Trajectory frames, reactants structure, and products structure.
    :rtype: tuple[list[Atoms], Atoms, Atoms]
    """
    hcho = molecule("H2CO")
    nh3 = molecule("NH3")
    nh3.translate([6.0, 0.0, 0.0])
    reactants = hcho + nh3
    amino = build_molecule_from_smiles("NCO", random_seed=42)
    symbols = reactants.get_chemical_symbols()
    products = reorder_atoms_to_symbols(amino, symbols)
    frames: list[Atoms] = [reactants.copy() for step_idx in range(5)] + [
        products.copy() for step_idx in range(5)
    ]
    return frames, reactants, products


def test_event_dataclass_equality() -> None:
    """
    Verify Event equality ignores Atoms instances and compares metadata.

    :return: None.
    :rtype: None
    """
    event_one = make_event(10.0, ("H", "O"))
    event_two = make_event(10.0, ("C", "N"))
    event_diff_time = make_event(20.0, ("H", "O"))
    assert event_one == event_two
    assert event_one != event_diff_time


def test_run_length_empty_and_basic() -> None:
    """
    Verify run_length on empty sequences and basic alternating states.

    :return: None.
    :rtype: None
    """
    assert run_length([]) == []
    runs = run_length(sequence_to_states("AAABBA"))
    expected = [
        StateRun(state=("A",), start=0, stop=3),
        StateRun(state=("B",), start=3, stop=5),
        StateRun(state=("A",), start=5, stop=6),
    ]
    assert runs == expected


def test_stable_runs_clean_transition() -> None:
    """
    Verify clean transition 'AAAAABBBBB' min 3 yields two stable runs.

    :return: None.
    :rtype: None
    """
    runs = stable_runs(sequence_to_states("AAAAABBBBB"), min_frames=3)
    assert len(runs) == 2
    assert runs[0] == StateRun(state=("A",), start=0, stop=5)
    assert runs[1] == StateRun(state=("B",), start=5, stop=10)


def test_stable_runs_flicker() -> None:
    """
    Verify flicker 'AAAABABAAAA' min 3 drops transients and merges to one run.

    :return: None.
    :rtype: None
    """
    runs = stable_runs(sequence_to_states("AAAABABAAAA"), min_frames=3)
    assert len(runs) == 1
    assert runs[0] == StateRun(state=("A",), start=0, stop=11)


def test_stable_runs_round_trip() -> None:
    """
    Verify round trip 'AAAABBBBAAAA' min 3 yields three stable runs.

    :return: None.
    :rtype: None
    """
    runs = stable_runs(sequence_to_states("AAAABBBBAAAA"), min_frames=3)
    assert len(runs) == 3
    assert runs[0] == StateRun(state=("A",), start=0, stop=4)
    assert runs[1] == StateRun(state=("B",), start=4, stop=8)
    assert runs[2] == StateRun(state=("A",), start=8, stop=12)


def test_stable_runs_short_first_run() -> None:
    """
    Verify short first run 'AABBBBB' min 3 is preserved as reactant start.

    :return: None.
    :rtype: None
    """
    runs = stable_runs(sequence_to_states("AABBBBB"), min_frames=3)
    assert len(runs) == 2
    assert runs[0] == StateRun(state=("A",), start=0, stop=2)
    assert runs[1] == StateRun(state=("B",), start=2, stop=7)


def test_stable_runs_middle_blip() -> None:
    """
    Verify middle blip 'AAAACBBBB' min 3 drops short blip between runs.

    :return: None.
    :rtype: None
    """
    runs = stable_runs(sequence_to_states("AAAACBBBB"), min_frames=3)
    assert len(runs) == 2
    assert runs[0] == StateRun(state=("A",), start=0, stop=4)
    assert runs[1] == StateRun(state=("B",), start=5, stop=9)


PureEventTuple = tuple[
    str | tuple[str, ...],
    str | tuple[str, ...],
    int,
    int,
]

PURE_STATE_CASES: list[tuple[str, int, list[PureEventTuple]]] = [
    # 1. Clean transition 'AAAAABBBBB' min 3 -> one event at index 5.
    ("AAAAABBBBB", 3, [(STATE_A, STATE_B, 4, 5)]),
    # 2. Flicker 'AAAABABAAAA' min 3 -> no event.
    ("AAAABABAAAA", 3, []),
    # 3. Round trip 'AAAABBBBAAAA' min 3 -> A->B at 4 and B->A at 8.
    (
        "AAAABBBBAAAA",
        3,
        [
            (STATE_A, STATE_B, 3, 4),
            (STATE_B, STATE_A, 7, 8),
        ],
    ),
    # 4. Short first run 'AABBBBB' min 3 -> first run kept, one event at 2.
    ("AABBBBB", 3, [(STATE_A, STATE_B, 1, 2)]),
    # 5. Middle blip 'AAAACBBBB' min 3 -> one event with frame_before index 3
    # and frame_after index 5.
    ("AAAACBBBB", 3, [(STATE_A, STATE_B, 3, 5)]),
]


@pytest.mark.parametrize(
    ("sequence", "min_frames", "expected_events"),
    PURE_STATE_CASES,
)
def test_find_events_pure_state_sequences(
    sequence: str,
    min_frames: int,
    expected_events: list[PureEventTuple],
) -> None:
    """
    Verify find_events extracts correct events across test patterns.

    :param sequence: Sequence string where letters select distinct elements.
    :type sequence: str
    :param min_frames: Minimum required duration in frames for stability.
    :type min_frames: int
    :param expected_events: Expected event descriptor tuples.
    :type expected_events: list[PureEventTuple]
    :return: None.
    :rtype: None
    """
    frames = make_dummy_frames(sequence)
    events = find_events(
        frames=frames, replicate=1, min_frames=min_frames, dump_fs=10.0
    )
    assert len(events) == len(expected_events)
    for event, expected in zip(events, expected_events, strict=True):
        exp_before, exp_after, before_idx, after_idx = expected
        assert event.replicate == 1
        assert event.time_fs == float(after_idx * 10.0)
        assert event.frame_before is frames[before_idx]
        assert event.frame_after is frames[after_idx]
        assert event.state_before == expected_state_tuple(exp_before)
        assert event.state_after == expected_state_tuple(exp_after)


def test_find_events_geometry_hcho_nh3() -> None:
    """
    Verify event extraction on embedded HCHO+NH3 to aminomethanol frames.

    :return: None.
    :rtype: None
    """
    frames, reactants, products = build_hcho_nh3_frames()
    events = find_events(frames=frames, replicate=2, min_frames=3, dump_fs=20.0)
    assert len(events) == 1
    event = events[0]
    assert event.replicate == 2
    assert event.time_fs == 100.0
    assert event.frame_before is frames[4]
    assert event.frame_after is frames[5]
    assert event.state_before == state_key(reactants)
    assert event.state_after == state_key(products)
    assert event.state_before != event.state_after


@pytest.mark.parametrize(
    ("sequence", "settle", "before_idx", "after_idx"),
    [
        # Endpoints step two frames into each run.
        ("AAAAAABBBBBB", 2, 3, 8),
        # A short first run keeps its first frame; the later run its last.
        ("AABBBBB", 3, 0, 5),
        # A dropped blip (C at 4) inside a merged run is skipped toward
        # the crossing, so the endpoint stays in the run's own state.
        ("AAAACAAAABBBB", 4, 5, 12),
    ],
)
def test_find_events_settle_frames(
    sequence: str,
    settle: int,
    before_idx: int,
    after_idx: int,
) -> None:
    """
    Verify settle_frames moves endpoints away from the crossing.

    :param sequence: Sequence string where letters select distinct elements.
    :type sequence: str
    :param settle: Frames to step away from the crossing.
    :type settle: int
    :param before_idx: Expected frame index of the earlier endpoint.
    :type before_idx: int
    :param after_idx: Expected frame index of the later endpoint.
    :type after_idx: int
    :return: None.
    :rtype: None
    """
    frames = make_dummy_frames(sequence)
    events = find_events(
        frames, 1, min_frames=3, dump_fs=10.0, settle_frames=settle
    )
    assert len(events) == 1
    assert events[0].frame_before is frames[before_idx]
    assert events[0].frame_after is frames[after_idx]
