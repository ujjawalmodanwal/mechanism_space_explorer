"""
Unit tests for transition state search pipeline and endpoint confirmation.

Verifies climbing-image NEB interpolation, Sella saddle refinement, finite
difference vibrational frequency verification, endpoint displacement checks,
and graceful handling of endpoint changes and unsupported species.
"""

from pathlib import Path

import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import CalculationFailed

from nrnet import ts_search
from nrnet.engines import EngineSettings, load_engine_settings
from nrnet.nanoreactor.candidates import Candidate, EndpointPair
from nrnet.ts_search import (
    SEARCH_STEPS,
    SearchSettings,
    SearchState,
    TSResult,
    TSRoute,
    TSSeed,
    TSStatus,
    load_search_settings,
    search_candidate,
    search_pair,
)


def build_formaldehyde() -> Atoms:
    """
    Construct planar formaldehyde H2CO in the YZ plane.

    :return: 3D atomic structure of formaldehyde.
    :rtype: Atoms
    """
    positions = [
        [0.0, 0.0, 0.6835],
        [0.0, 0.0, -0.5366],
        [0.0, 0.9344, -1.1242],
        [0.0, -0.9344, -1.1242],
    ]
    return Atoms(symbols=["O", "C", "H", "H"], positions=positions)


def build_hydroxycarbene() -> Atoms:
    """
    Construct planar trans-hydroxycarbene HCOH aligned in the YZ plane.

    :return: 3D atomic structure of trans-hydroxycarbene.
    :rtype: Atoms
    """
    positions = [
        [0.0, -0.011, 0.612],
        [0.0, -0.022, -0.693],
        [0.0, 1.051, -0.993],
        [0.0, -0.917, 0.974],
    ]
    return Atoms(symbols=["O", "C", "H", "H"], positions=positions)


def build_collapsing_product() -> Atoms:
    """
    Construct perturbed geometry that relaxes back into formaldehyde.

    :return: 3D atomic structure with unstable H displacement.
    :rtype: Atoms
    """
    positions = [
        [0.0, 0.0, 0.65],
        [0.0, 0.0, -0.65],
        [0.0, 0.93, -1.12],
        [0.0, -0.5, 0.2],
    ]
    return Atoms(symbols=["O", "C", "H", "H"], positions=positions)


def build_test_pair() -> EndpointPair:
    """
    Construct valid H2CO -> trans-HCOH 1,2-H shift endpoint pair.

    :return: Configured EndpointPair dataclass.
    :rtype: EndpointPair
    """
    return EndpointPair(
        replicate=0,
        time_fs=100.0,
        atom_ids=(0, 1, 2, 3),
        before=build_formaldehyde(),
        after=build_hydroxycarbene(),
    )


@pytest.fixture
def xtb_engine_settings() -> EngineSettings:
    """
    Load xTB engine settings fixture.

    :return: Configured EngineSettings dataclass.
    :rtype: EngineSettings
    """
    return load_engine_settings("configs/engines.yaml", "xtb")


@pytest.fixture
def default_search_settings() -> SearchSettings:
    """
    Load default search settings fixture.

    :return: Configured SearchSettings dataclass.
    :rtype: SearchSettings
    """
    return load_search_settings("configs/search.yaml")


@pytest.fixture
def closed_engine_settings() -> EngineSettings:
    """
    Construct closed-shell only engine settings fixture.

    :return: Closed-shell EngineSettings.
    :rtype: EngineSettings
    """
    return EngineSettings("xtb", "GFN2-xTB", "cpu", closed_shell_only=True)


def build_collapsing_pair() -> EndpointPair:
    """
    Construct endpoint pair that collapses back to reactant.

    :return: Configured EndpointPair dataclass.
    :rtype: EndpointPair
    """
    return EndpointPair(
        replicate=1,
        time_fs=200.0,
        atom_ids=(0, 1, 2, 3),
        before=build_formaldehyde(),
        after=build_collapsing_product(),
    )


def build_test_candidate() -> Candidate:
    """
    Construct test Candidate reaction with two endpoint pairs.

    :return: Candidate reaction dataclass.
    :rtype: Candidate
    """
    pair_bad = EndpointPair(
        2, 300.0, (0, 1, 2, 3), build_formaldehyde(), build_collapsing_product()
    )
    return Candidate(
        reaction_id="R_TEST",
        state_a=("CH2O-021b6208",),
        state_b=("CH2O-63452dc2",),
        reactant_keys=("CH2O-021b6208",),
        product_keys=("CH2O-63452dc2",),
        pairs=(build_test_pair(), pair_bad),
        mapped_smiles="[O:1]=[C:2]([H:3])[H:4]>>[O:1]([H:4])-[C:2]-[H:3]",
        n_events=2,
        first_seen_fs=100.0,
    )


def assert_valid_confirmed_ts(result: TSResult) -> None:
    """
    Assert valid confirmed transition state result attributes.

    :param result: Transition state search result.
    :type result: TSResult
    :return: None.
    :rtype: None
    """
    assert isinstance(result, TSResult)
    assert result.status == TSStatus.confirmed
    assert result.ts_atoms is not None
    assert result.neb_converged is True
    assert result.imaginary_cm is not None
    assert result.imaginary_cm < -500.0
    assert result.vibrational_cm is not None
    assert len(result.vibrational_cm) == 5
    assert all(freq > 0.0 for freq in result.vibrational_cm)
    assert result.ts_energy_ev is not None
    assert result.reactant_energy_ev is not None
    barrier_ev = result.ts_energy_ev - result.reactant_energy_ev
    assert 2.0 <= barrier_ev <= 5.0
    assert result.force_calls > 0


def test_load_search_settings() -> None:
    """
    Verify the search configuration loads with consistent thresholds.

    :return: None.
    :rtype: None
    """
    settings = load_search_settings("configs/search.yaml")
    assert isinstance(settings, SearchSettings)
    assert settings.neb_images > 2
    assert settings.neb_relax_fmax_ev_per_ang > settings.neb_fmax_ev_per_ang
    assert settings.neb_fmax_ev_per_ang > settings.saddle_fmax_ev_per_ang
    assert settings.imaginary_threshold_cm < 0.0
    assert settings.confirm_step_ang > 0.0


def test_load_search_settings_rejects_missing_key(tmp_path: Path) -> None:
    """
    Verify a search section missing a threshold fails instead of defaulting.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    path = tmp_path / "search.yaml"
    path.write_text("search:\n  neb_images: 13\n")
    with pytest.raises(TypeError):
        load_search_settings(path)


def test_ts_search_formaldehyde_to_hydroxycarbene(
    xtb_engine_settings: EngineSettings,
    default_search_settings: SearchSettings,
) -> None:
    """
    Verify transition state search for formaldehyde 1,2-H shift.

    :param xtb_engine_settings: xTB computational engine configuration fixture.
    :type xtb_engine_settings: EngineSettings
    :param default_search_settings: Default TS search configuration fixture.
    :type default_search_settings: SearchSettings
    :return: None.
    :rtype: None
    """
    result = search_pair(
        pair=build_test_pair(),
        pair_index=0,
        engine_settings=xtb_engine_settings,
        search_settings=default_search_settings,
        seed=None,
    )
    assert_valid_confirmed_ts(result)
    assert result.vibrational_cm is not None
    assert len(result.vibrational_cm) == 5
    assert all(freq > 0.0 for freq in result.vibrational_cm)


def test_ts_search_endpoint_changed(
    xtb_engine_settings: EngineSettings,
    default_search_settings: SearchSettings,
) -> None:
    """
    Verify endpoint_changed status when product relaxes back to reactant.

    :param xtb_engine_settings: xTB computational engine configuration fixture.
    :type xtb_engine_settings: EngineSettings
    :param default_search_settings: Default TS search configuration fixture.
    :type default_search_settings: SearchSettings
    :return: None.
    :rtype: None
    """
    result = search_pair(
        pair=build_collapsing_pair(),
        pair_index=0,
        engine_settings=xtb_engine_settings,
        search_settings=default_search_settings,
        seed=None,
    )
    assert result.status == TSStatus.endpoint_changed
    assert result.ts_atoms is None


def test_ts_search_unsupported_species(
    closed_engine_settings: EngineSettings,
    default_search_settings: SearchSettings,
) -> None:
    """
    Verify unsupported status when open-shell species given to closed engine.

    :param closed_engine_settings: Closed-shell engine configuration fixture.
    :type closed_engine_settings: EngineSettings
    :param default_search_settings: Default TS search configuration fixture.
    :type default_search_settings: SearchSettings
    :return: None.
    :rtype: None
    """
    doublet_mol = molecule("CH3")
    pair = EndpointPair(0, 0.0, (0, 1, 2, 3), doublet_mol, doublet_mol)
    result = search_pair(
        pair=pair,
        pair_index=0,
        engine_settings=closed_engine_settings,
        search_settings=default_search_settings,
        seed=None,
    )
    assert result.status == TSStatus.unsupported
    assert result.ts_atoms is None


def raise_scf_failure(state: SearchState) -> TSStatus | None:
    """
    Stand in for a search step whose engine call fails.

    :param state: Active search state, unused.
    :type state: SearchState
    :return: Never returns.
    :rtype: TSStatus or None
    :raises CalculationFailed: Always, with an SCF message.
    """
    raise CalculationFailed("SCF not converged in 250 cycles")


def test_engine_error_keeps_message(
    monkeypatch: pytest.MonkeyPatch,
    xtb_engine_settings: EngineSettings,
    default_search_settings: SearchSettings,
) -> None:
    """
    Verify an engine failure is reported with the engine's own message.

    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    :param xtb_engine_settings: xTB engine configuration fixture.
    :type xtb_engine_settings: EngineSettings
    :param default_search_settings: Default TS search configuration fixture.
    :type default_search_settings: SearchSettings
    :return: None.
    :rtype: None
    """
    monkeypatch.setattr(ts_search, "SEARCH_STEPS", (raise_scf_failure,))
    result = search_pair(
        build_test_pair(), 0, xtb_engine_settings, default_search_settings, None
    )
    assert result.status == TSStatus.engine_error
    assert result.note == "SCF not converged in 250 cycles"


def test_search_candidate_stops_at_first_confirmed(
    xtb_engine_settings: EngineSettings,
    default_search_settings: SearchSettings,
) -> None:
    """
    Verify search_candidate tries pairs in order and stops on first confirmed.

    :param xtb_engine_settings: xTB computational engine configuration fixture.
    :type xtb_engine_settings: EngineSettings
    :param default_search_settings: Default TS search configuration fixture.
    :type default_search_settings: SearchSettings
    :return: None.
    :rtype: None
    """
    results = search_candidate(
        candidate=build_test_candidate(),
        engine_settings=xtb_engine_settings,
        search_settings=default_search_settings,
        seed=None,
    )
    assert len(results) == 1
    assert results[0].status == TSStatus.confirmed


def test_search_state_and_steps() -> None:
    """
    Verify SearchState initialization and SEARCH_STEPS sequence.

    :return: None.
    :rtype: None
    """
    pair = build_test_pair()
    engine_settings = load_engine_settings("configs/engines.yaml", "xtb")
    search_settings = load_search_settings("configs/search.yaml")
    state = SearchState(pair, 0, engine_settings, search_settings)
    assert len(state.expected_keys) == 2
    assert len(SEARCH_STEPS) == 5
    assert state.force_calls == 0
    assert state.vibrational_cm is None
    assert state.mode is None


def test_seeded_search_confirms_from_band_ts(
    xtb_engine_settings: EngineSettings,
    default_search_settings: SearchSettings,
) -> None:
    """
    Verify a confirmed TS used as a seed is confirmed again, more cheaply.

    :param xtb_engine_settings: xTB computational engine configuration fixture.
    :type xtb_engine_settings: EngineSettings
    :param default_search_settings: Default TS search configuration fixture.
    :type default_search_settings: SearchSettings
    :return: None.
    :rtype: None
    """
    candidate = build_test_candidate()
    settings = (xtb_engine_settings, default_search_settings)
    band = search_candidate(candidate, *settings, None)[-1]
    assert band.ts_atoms is not None
    seed = TSSeed(band.pair_index, band.ts_atoms, "xtb")
    results = search_candidate(candidate, *settings, seed)
    assert len(results) == 1
    assert results[0].status == TSStatus.confirmed
    assert results[0].route == TSRoute.seed
    assert results[0].force_calls < band.force_calls


def fake_search_pair(
    pair: EndpointPair,
    pair_index: int,
    engine_settings: EngineSettings,
    search_settings: SearchSettings,
    seed: Atoms | None,
) -> TSResult:
    """
    Stand in for search_pair: a seeded attempt fails, a band attempt works.

    :param pair: Endpoint pair, unused.
    :type pair: EndpointPair
    :param pair_index: Index of the pair.
    :type pair_index: int
    :param engine_settings: Engine settings, unused.
    :type engine_settings: EngineSettings
    :param search_settings: Search settings, unused.
    :type search_settings: SearchSettings
    :param seed: Seed geometry or None.
    :type seed: Atoms or None
    :return: endpoint_mismatch for a seed, confirmed for a band.
    :rtype: TSResult
    """
    if seed is None:
        return TSResult(TSStatus.confirmed, pair_index, route=TSRoute.band)
    return TSResult(TSStatus.endpoint_mismatch, pair_index, route=TSRoute.seed)


def test_failed_seed_falls_back_to_band(
    monkeypatch: pytest.MonkeyPatch,
    xtb_engine_settings: EngineSettings,
    default_search_settings: SearchSettings,
) -> None:
    """
    Verify a failed seeded attempt is followed by the engine's own band.

    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    :param xtb_engine_settings: xTB computational engine configuration fixture.
    :type xtb_engine_settings: EngineSettings
    :param default_search_settings: Default TS search configuration fixture.
    :type default_search_settings: SearchSettings
    :return: None.
    :rtype: None
    """
    monkeypatch.setattr(ts_search, "search_pair", fake_search_pair)
    seed = TSSeed(0, build_formaldehyde(), "xtb")
    results = search_candidate(
        build_test_candidate(),
        xtb_engine_settings,
        default_search_settings,
        seed,
    )
    assert [res.route for res in results] == [TSRoute.seed, TSRoute.band]
    assert results[-1].status == TSStatus.confirmed
