"""
Unit tests for candidate reaction extraction, filtering, and serialization.

Verifies end-to-end freezing of reaction trajectories, spectator atom
exclusion, rejection of identical-state events, round-trip serialization,
and manifest integrity checking.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from rdkit import Chem
from rdkit.Chem import AllChem

from nrnet.chem import state_key
from nrnet.engines import load_engine_settings
from nrnet.nanoreactor.candidate_store import freeze, load_candidates
from nrnet.nanoreactor.candidates import (
    Candidate,
    EndpointPair,
    Species,
    build_endpoint_pair,
    compile_candidate_stats,
    group_by_state_pair,
)
from nrnet.nanoreactor.events import Event


def build_aminomethanol() -> Atoms:
    """
    Build aminomethanol with atom ordering matching HCHO + NH3.

    :return: 3D embedded aminomethanol Atoms instance.
    :rtype: Atoms
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    smiles = "[O:1]([H:8])[C:2]([H:3])([H:4])[N:5]([H:6])[H:7]"
    mol = Chem.MolFromSmiles(smiles, params)
    map_to_idx = {
        atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()
    }
    order = [map_to_idx[idx + 1] for idx in range(len(mol.GetAtoms()))]
    renum = Chem.RenumberAtoms(mol, order)
    AllChem.EmbedMolecule(renum, randomSeed=42)
    AllChem.UFFOptimizeMolecule(renum)
    positions = renum.GetConformer().GetPositions()
    symbols = [atom.GetSymbol() for atom in renum.GetAtoms()]
    return Atoms(symbols=symbols, positions=positions)


def build_reactants() -> Atoms:
    """
    Construct separated HCHO and NH3 reactants configuration.

    :return: Combined reactants Atoms object.
    :rtype: Atoms
    """
    hcho = molecule("H2CO")
    nh3 = molecule("NH3")
    nh3.translate([4.0, 0.0, 0.0])
    return hcho + nh3


def build_spectator_water() -> Atoms:
    """
    Construct spectator water molecule translated 8 Angstroms away.

    :return: Translated water Atoms object.
    :rtype: Atoms
    """
    water = molecule("H2O")
    water.translate([0.0, 0.0, 8.0])
    return water


def build_reaction_trajectory() -> tuple[Atoms, list[list[Atoms]]]:
    """
    Build test trajectory containing HCHO + NH3 -> aminomethanol transition.

    :return: Initial reactant structure and single-replicate frame list.
    :rtype: tuple[Atoms, list[list[Atoms]]]
    """
    reactants = build_reactants()
    products = build_aminomethanol()
    spectator = build_spectator_water()
    before_frame = reactants + spectator
    after_frame = products + spectator
    frames = [before_frame.copy() for _ in range(5)] + [
        after_frame.copy() for _ in range(5)
    ]
    return before_frame, [frames]


def assert_candidate_keys(cand: Candidate) -> None:
    """
    Assert reactant, product keys and reacting atoms on candidate.

    :param cand: Candidate reaction to inspect.
    :type cand: Candidate
    :return: None.
    :rtype: None
    """
    assert cand.reaction_id == "R001"
    assert any(k.startswith("CH2O") for k in cand.reactant_keys)
    assert any(k.startswith("H3N") for k in cand.reactant_keys)
    assert any(k.startswith("CH5NO") for k in cand.product_keys)
    assert cand.pairs[0].atom_ids == tuple(range(8))


def test_hcho_nh3_freezes_one_candidate_with_spectator(
    tmp_path: Path,
) -> None:
    """
    Verify HCHO+NH3->aminomethanol freezes with spectator water excluded.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    reactant, trajectories = build_reaction_trajectory()
    cset = freeze(
        reactant=reactant,
        trajectories=trajectories,
        min_frames=3,
        dump_fs=10.0,
        engine_settings="configs/engines.yaml",
        out_dir=tmp_path,
    )
    assert len(cset.candidates) == 1
    assert_candidate_keys(cset.candidates[0])
    assert len(cset.species) == 4


def test_rejected_event_with_same_state() -> None:
    """
    Verify event whose before and after share the same state is rejected.

    :return: None.
    :rtype: None
    """
    reactant = build_reactants()
    dummy_event = Event(
        replicate=0,
        time_fs=10.0,
        frame_before=reactant,
        frame_after=reactant,
        state_before=state_key(reactant),
        state_after=state_key(reactant),
    )
    assert len(group_by_state_pair([dummy_event])) == 0
    settings = load_engine_settings("configs/engines.yaml", "xtb")
    pair, reason = build_endpoint_pair(dummy_event, settings)
    assert pair is None
    assert reason in ("no_reacting_atoms", "identical_states")


def assert_pair_round_trip(orig: EndpointPair, loaded: EndpointPair) -> None:
    """
    Assert equality of endpoint pair metadata and atomic coordinates.

    :param orig: Original EndpointPair instance.
    :type orig: EndpointPair
    :param loaded: Reconstructed EndpointPair instance.
    :type loaded: EndpointPair
    :return: None.
    :rtype: None
    """
    assert orig.atom_ids == loaded.atom_ids
    assert orig.replicate == loaded.replicate
    assert orig.time_fs == loaded.time_fs
    pos_b = orig.before.get_positions()
    pos_a = orig.after.get_positions()
    assert np.allclose(pos_b, loaded.before.get_positions(), atol=1e-6)
    assert np.allclose(pos_a, loaded.after.get_positions(), atol=1e-6)


def assert_candidate_round_trip(orig: Candidate, loaded: Candidate) -> None:
    """
    Assert equality of candidate reaction metadata and contained pairs.

    :param orig: Original Candidate instance.
    :type orig: Candidate
    :param loaded: Reconstructed Candidate instance.
    :type loaded: Candidate
    :return: None.
    :rtype: None
    """
    assert orig.reaction_id == loaded.reaction_id
    assert orig.state_a == loaded.state_a
    assert orig.state_b == loaded.state_b
    assert orig.reactant_keys == loaded.reactant_keys
    assert orig.product_keys == loaded.product_keys
    assert orig.mapped_smiles == loaded.mapped_smiles
    assert orig.n_events == loaded.n_events
    assert orig.first_seen_fs == loaded.first_seen_fs
    assert len(orig.pairs) == len(loaded.pairs)
    assert_pair_round_trip(orig.pairs[0], loaded.pairs[0])


def assert_species_round_trip(orig: Species, loaded: Species) -> None:
    """
    Assert equality of species metadata and atomic coordinates.

    :param orig: Original Species instance.
    :type orig: Species
    :param loaded: Reconstructed Species instance.
    :type loaded: Species
    :return: None.
    :rtype: None
    """
    assert orig.key == loaded.key
    assert orig.smiles == loaded.smiles
    assert orig.inchikey == loaded.inchikey
    assert orig.multiplicity == loaded.multiplicity
    pos_orig = orig.atoms.get_positions()
    assert np.allclose(pos_orig, loaded.atoms.get_positions(), atol=1e-6)


def test_freeze_load_round_trip(tmp_path: Path) -> None:
    """
    Verify freeze then load round-trips identifiers, keys, and coordinates.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    reactant, trajectories = build_reaction_trajectory()
    orig = freeze(
        reactant, trajectories, 3, 10.0, "configs/engines.yaml", tmp_path
    )
    loaded = load_candidates(tmp_path)
    assert orig.reactant_state == loaded.reactant_state
    assert orig.stats == loaded.stats
    assert orig.stats["events"] == 1
    assert orig.stats["event_groups"] == 1
    assert orig.stats["candidates"] == 1
    assert orig.stats["species"] == 4
    assert len(orig.candidates) == len(loaded.candidates)
    assert_candidate_round_trip(orig.candidates[0], loaded.candidates[0])
    for key, spec in orig.species.items():
        assert_species_round_trip(spec, loaded.species[key])


def test_compile_candidate_stats_with_rejections() -> None:
    """
    Verify candidate set stats compilation with rejection reasons.

    :return: None.
    :rtype: None
    """
    rejections: Counter[str] = Counter({"unconverged": 2, "state_mismatch": 1})
    stats = compile_candidate_stats([], [], [], {}, rejections)
    assert stats["events"] == 0
    assert stats["event_groups"] == 0
    assert stats["candidates"] == 0
    assert stats["species"] == 0
    assert stats["rejected_unconverged"] == 2
    assert stats["rejected_state_mismatch"] == 1


def test_tampered_manifest_raises(tmp_path: Path) -> None:
    """
    Verify modifying candidates.json causes load_candidates to raise ValueError.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    reactant, trajectories = build_reaction_trajectory()
    freeze(
        reactant=reactant,
        trajectories=trajectories,
        min_frames=3,
        dump_fs=10.0,
        engine_settings="configs/engines.yaml",
        out_dir=tmp_path,
    )
    cand_path = tmp_path / "candidates.json"
    content = cand_path.read_bytes()
    cand_path.write_bytes(content + b" ")
    with pytest.raises(ValueError, match=r"candidates\.json"):
        load_candidates(tmp_path)
