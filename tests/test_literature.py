"""
Unit tests for hand-built literature reaction ingestion and candidate sets.

Validates literature reactions loading, symbol consistency checking, candidate
matching, pair replacement, source tagging, and graph edge provenance.
"""

from dataclasses import replace
from pathlib import Path

import ase.io
import pytest
from ase import Atoms

from nrnet.cli import freeze_system_candidates
from nrnet.evaluate import reaction_energies
from nrnet.literature import (
    LiteratureReaction,
    add_literature_to_candidates,
    load_literature_reactions,
)
from nrnet.nanoreactor.candidate_store import load_candidates
from nrnet.nanoreactor.candidates import (
    Candidate,
    CandidateSet,
    EndpointPair,
)
from nrnet.nanoreactor.md import load_md_settings
from nrnet.network import ReactionBarrier, build_network
from nrnet.refine import reaction_meta_fields
from nrnet.ts_search import TSResult, TSRoute, TSStatus


def test_literature_loader_good_file() -> None:
    """
    Verify loading good literature.yaml returns valid LiteratureReactions.

    :return: None.
    :rtype: None
    """
    rxns = load_literature_reactions("systems/hcooh/literature.yaml")
    assert len(rxns) == 1
    rxn = rxns[0]
    assert rxn.name == "decarboxylation"
    assert len(rxn.reactant) == 5
    assert len(rxn.product) == 5
    assert "HCOOH" in rxn.note
    r_syms = rxn.reactant.get_chemical_symbols()
    p_syms = rxn.product.get_chemical_symbols()
    assert r_syms == p_syms


def test_literature_loader_symbol_mismatch(tmp_path: Path) -> None:
    """
    Verify ValueError is raised when chemical symbols do not match.

    :param tmp_path: Temporary filesystem path fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    (tmp_path / "r.xyz").write_text("1\ncomment\nC 0 0 0\n")
    (tmp_path / "p.xyz").write_text("1\ncomment\nN 0 0 0\n")
    yaml_content = (
        "reactions:\n"
        "  - name: mismatch_rxn\n"
        "    reactant: r.xyz\n"
        "    product: p.xyz\n"
    )
    (tmp_path / "literature.yaml").write_text(yaml_content)
    with pytest.raises(ValueError, match="symbols mismatch"):
        load_literature_reactions(tmp_path / "literature.yaml")


def test_add_literature_new_candidate() -> None:
    """
    Verify adding literature reaction creates new flagged candidate.

    :return: None.
    :rtype: None
    """
    cset = CandidateSet(
        reactant_state=("CH2O2-2a71d22e",),
        species={},
        candidates=(),
        stats={},
    )
    rxns = load_literature_reactions("systems/hcooh/literature.yaml")
    updated = add_literature_to_candidates(cset, rxns, "configs/engines.yaml")
    assert len(updated.candidates) == 1
    cand = updated.candidates[0]
    assert cand.source == "literature"
    assert cand.literature_name == "decarboxylation"
    assert cand.n_events == 0
    assert cand.pairs[0].replicate == 0
    assert len(cand.pairs) == 1


def test_add_literature_hcooh_h2o() -> None:
    """
    Verify adding literature reaction for hcooh_h2o creates candidate.

    :return: None.
    :rtype: None
    """
    cset = CandidateSet(
        reactant_state=("CH2O2-2a71d22e", "H2O-2ebfe80f"),
        species={},
        candidates=(),
        stats={},
    )
    rxns = load_literature_reactions("systems/hcooh_h2o/literature.yaml")
    updated = add_literature_to_candidates(cset, rxns, "configs/engines.yaml")
    assert len(updated.candidates) == 1
    cand = updated.candidates[0]
    assert cand.source == "literature"
    assert cand.literature_name == "decarboxylation_water_assisted"
    assert cand.pairs[0].atom_ids == tuple(range(8))


def make_dummy_pair() -> EndpointPair:
    """
    Construct a dummy endpoint pair for testing.

    :return: Dummy EndpointPair instance.
    :rtype: EndpointPair
    """
    return EndpointPair(
        replicate=1,
        time_fs=100.0,
        atom_ids=(0, 1, 2, 3, 4),
        before=Atoms("CH2O2"),
        after=Atoms("CH2O2"),
    )


def make_dummy_hcooh_candidate() -> Candidate:
    """
    Construct a dummy candidate reaction for HCOOH decarboxylation.

    :return: Candidate dataclass instance.
    :rtype: Candidate
    """
    return Candidate(
        reaction_id="R001",
        state_a=("CH2O2-2a71d22e",),
        state_b=("CO2-9d2ea55c", "H2-212475fb"),
        reactant_keys=("CH2O2-2a71d22e",),
        product_keys=("CO2-9d2ea55c", "H2-212475fb"),
        pairs=(make_dummy_pair(),),
        mapped_smiles="old_smiles",
        n_events=5,
        first_seen_fs=100.0,
        source="md",
    )


def make_existing_hcooh_set() -> CandidateSet:
    """
    Construct test CandidateSet with existing decarboxylation candidate.

    :return: CandidateSet containing dummy decarboxylation candidate.
    :rtype: CandidateSet
    """
    return CandidateSet(
        ("CH2O2-2a71d22e",), {}, (make_dummy_hcooh_candidate(),), {}
    )


def test_add_literature_matched_candidate() -> None:
    """
    Verify a matched literature reaction is only labelled, not added.

    :return: None.
    :rtype: None
    """
    cset = make_existing_hcooh_set()
    rxns = load_literature_reactions("systems/hcooh/literature.yaml")
    updated = add_literature_to_candidates(cset, rxns, "configs/engines.yaml")
    assert len(updated.candidates) == 1
    cand = updated.candidates[0]
    assert cand.source == "md"
    assert cand.literature_name == "decarboxylation"
    assert cand.n_events == 5
    assert [pair.replicate for pair in cand.pairs] == [1]
    assert cand.mapped_smiles == "old_smiles"


def test_bystander_water_does_not_match_assisted_reaction() -> None:
    """
    Verify an MD decarboxylation beside a spectator water is not taken for
    the water-relay reaction, which shares its whole states (F12).

    :return: None.
    :rtype: None
    """
    water = "H2O-2ebfe80f"
    bystander = replace(
        make_dummy_hcooh_candidate(),
        state_a=("CH2O2-2a71d22e", water),
        state_b=("CO2-9d2ea55c", "H2-212475fb", water),
    )
    cset = CandidateSet(
        reactant_state=("CH2O2-2a71d22e", water),
        species={},
        candidates=(bystander,),
        stats={},
    )
    rxns = load_literature_reactions("systems/hcooh_h2o/literature.yaml")
    updated = add_literature_to_candidates(cset, rxns, "configs/engines.yaml")
    assert len(updated.candidates) == 2
    assert updated.candidates[0].literature_name is None
    added = updated.candidates[1]
    assert added.source == "literature"
    assert added.literature_name == "decarboxylation_water_assisted"
    assert water in added.reactant_keys


def test_literature_same_state_error() -> None:
    """
    Verify ValueError is raised if literature reaction endpoints share state.

    :return: None.
    :rtype: None
    """
    rxns = load_literature_reactions("systems/hcooh/literature.yaml")
    rxn = rxns[0]
    same_rxn = LiteratureReaction(
        name="identity_rxn",
        reactant=rxn.reactant,
        product=rxn.reactant,
    )
    cset = CandidateSet(("CH2O2-2a71d22e",), {}, (), {})
    with pytest.raises(ValueError, match=r"identical state|same state"):
        add_literature_to_candidates(cset, [same_rxn], "configs/engines.yaml")


def test_old_candidates_json_loads_with_defaults() -> None:
    """
    Verify legacy candidates.json loads with default source and literature_name.

    :return: None.
    :rtype: None
    """
    cset = load_candidates("tests/data/hcho_nh3_candidates")
    assert len(cset.candidates) == 4
    for cand in cset.candidates:
        assert cand.source == "md"
        assert cand.literature_name is None


def make_flagged_candidate() -> Candidate:
    """
    Construct a dummy Candidate instance with literature provenance.

    :return: Populated Candidate dataclass instance.
    :rtype: Candidate
    """
    return Candidate(
        reaction_id="R001",
        state_a=("A",),
        state_b=("B",),
        reactant_keys=("A",),
        product_keys=("B",),
        pairs=(),
        mapped_smiles="",
        n_events=0,
        first_seen_fs=0.0,
        source="literature",
        literature_name="decarboxylation",
    )


def test_refine_row_carries_flag() -> None:
    """
    Verify candidate source and literature_name propagate to refine record.

    :return: None.
    :rtype: None
    """
    cand = make_flagged_candidate()
    res = TSResult(pair_index=0, status=TSStatus.confirmed, route=TSRoute.band)
    row = reaction_meta_fields(cand, res, None)
    assert row["source"] == "literature"
    assert row["literature_name"] == "decarboxylation"
    e1_row = reaction_energies(cand, None, {}, {}, "298K")
    assert e1_row["source"] == "literature"


def test_network_edge_carries_flag() -> None:
    """
    Verify network edges in both directions carry source and literature name.

    :return: None.
    :rtype: None
    """
    cand = make_flagged_candidate()
    barrier = ReactionBarrier(
        "A", "B", 0.5, source=cand.source, literature_name=cand.literature_name
    )
    graph = build_network({"A": 0.0, "B": 0.0}, [barrier], 298.15)
    assert graph["A"]["B"]["source"] == "literature"
    assert graph["A"]["B"]["literature_name"] == "decarboxylation"
    assert graph["B"]["A"]["source"] == "literature"
    assert graph["B"]["A"]["literature_name"] == "decarboxylation"


def run_hcooh_freeze_with_lit(
    tmp_path: Path,
) -> tuple[CandidateSet, CandidateSet]:
    """
    Execute freeze_system_candidates with literature and reload results.

    :param tmp_path: Destination candidate directory.
    :type tmp_path: Path
    :return: Tuple of returned and reloaded CandidateSet instances.
    :rtype: tuple[CandidateSet, CandidateSet]
    """
    sys_dir = Path("systems/hcooh")
    reactant = ase.io.read(sys_dir / "reactant_trans.xyz")
    settings = load_md_settings(sys_dir / "system.yaml")
    cset = freeze_system_candidates(
        reactant,
        settings,
        [[reactant.copy() for _ in range(5)]],
        "configs/engines.yaml",
        tmp_path,
        add_literature=True,
        system_dir=sys_dir,
    )
    return cset, load_candidates(tmp_path)


def test_freeze_system_candidates_with_literature(tmp_path: Path) -> None:
    """
    Verify freeze_system_candidates integrates literature reactions into files.

    :param tmp_path: Temporary filesystem path fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    cset, loaded = run_hcooh_freeze_with_lit(tmp_path)
    assert any(c.literature_name == "decarboxylation" for c in cset.candidates)
    assert any(
        c.literature_name == "decarboxylation" for c in loaded.candidates
    )
