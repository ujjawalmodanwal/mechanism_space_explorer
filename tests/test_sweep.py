"""
Unit tests for parameter sweep execution, metrics logging, and candidate freeze.

Tests cover start structure optimization with wall constraints, parameter grid
expansion, multiprocessing pool task assembly, trajectory resume skipping,
literature product matching, and CSV metrics updating.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import ase.io
import pytest
from ase import Atoms
from ase.build import molecule
from rdkit import Chem

from nrnet.cli import (
    freeze_system_candidates,
    load_from_runs_trajectories,
)
from nrnet.literature import (
    LiteratureProduct,
    candidate_set_holds_product,
    load_literature_products,
)
from nrnet.nanoreactor.candidates import Candidate, CandidateSet, Species
from nrnet.nanoreactor.md import (
    MDSettings,
    load_md_settings,
    run_optimization,
)
from nrnet.sweep import (
    SweepSetting,
    build_base_metrics,
    build_sweep_records,
    build_sweep_row,
    collect_sweep_replicate_tasks,
    compute_group_overlap,
    count_failed_replicates,
    expand_sweep_grid,
    is_group_complete,
    merge_sweep_csv_rows,
    read_existing_csv_rows,
    sweep_group_dir,
    update_sweep_candidates_json,
    update_sweep_csv,
)


def create_test_md_settings(**kwargs: Any) -> MDSettings:
    """
    Construct sample MDSettings instance for testing using default config.

    :param kwargs: Optional keyword argument overrides.
    :type kwargs: Any
    :return: Configured MDSettings instance.
    :rtype: MDSettings
    """
    base = load_md_settings("systems/hcho_nh3/system.yaml")
    return replace(base, **kwargs)


@contextmanager
def mock_opt_session(
    workdir: Path,
    opt_atoms: Atoms,
) -> Iterator[MagicMock]:
    """
    Mock xTB process session writing xtbopt.xyz into workdir.

    :param workdir: Execution directory where xtbopt.xyz is written.
    :type workdir: Path
    :param opt_atoms: Molecular geometry to write as optimization output.
    :type opt_atoms: Atoms
    :return: Mock process context.
    :rtype: Iterator[MagicMock]
    """
    ase.io.write(str(workdir / "xtbopt.xyz"), opt_atoms, format="xyz")
    proc = MagicMock()
    proc.communicate.return_value = ("", "")
    proc.returncode = 0
    yield proc


def test_opt_deck_and_valid_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Verify optimization control input holds only wall and preserves state.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    :return: None.
    :rtype: None
    """
    mol = molecule("H2O")
    settings = create_test_md_settings()

    def patch_session(_cmd: Any, wdir: Path) -> Iterator[MagicMock]:
        return mock_opt_session(wdir, mol)

    monkeypatch.setattr(
        "nrnet.nanoreactor.md.xtb_process_session", patch_session
    )
    out = run_optimization(mol, settings, 0, tmp_path)
    opt_dir = tmp_path / "opt_0"
    control = (opt_dir / "control.inp").read_text(encoding="utf-8")
    assert "$wall" in control and "$md" not in control
    assert (opt_dir / "start_optimized.xyz").is_file()
    assert len(out) == len(mol)


def test_opt_rejects_changed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Verify start optimization raises ValueError when chemical state changes.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    :return: None.
    :rtype: None
    """
    mol = molecule("H2O")
    dissociated = molecule("H2O")
    dissociated.positions[1] += [10.0, 0.0, 0.0]
    settings = create_test_md_settings()

    def patch_session(_cmd: Any, wdir: Path) -> Iterator[MagicMock]:
        return mock_opt_session(wdir, dissociated)

    monkeypatch.setattr(
        "nrnet.nanoreactor.md.xtb_process_session", patch_session
    )
    with pytest.raises(ValueError, match="differs from start"):
        run_optimization(mol, settings, 0, tmp_path)


def test_grid_expansion_and_kpush_arithmetic() -> None:
    """
    Verify sweep grid expansion and per-atom kpush arithmetic rounding.

    :return: None.
    :rtype: None
    """
    temps = [300.0, 600.0]
    k_per_atom = [0.015, 0.02, 0.025, 0.03]
    grid = expand_sweep_grid(
        temperatures=temps,
        kpush_per_atom=k_per_atom,
        groups=2,
        first_group=1,
        n_atoms=8,
    )
    assert len(grid) == 16
    assert all(s.kpush_eh == round(s.kpush_per_atom * 8, 4) for s in grid)
    assert {s.group for s in grid} == {1, 2}
    assert {s.temperature_k for s in grid} == {300.0, 600.0}
    assert grid[0].kpush_eh == 0.12


def test_one_pool_task_list_covers_every_setting(tmp_path: Path) -> None:
    """
    Verify replicate task generation covers all settings, starts, and groups.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    starts = [("s0", molecule("H2O")), ("s1", molecule("H2O"))]
    settings = create_test_md_settings(replicates=2, base_seed=42)
    grid = expand_sweep_grid(
        [300.0, 500.0], [0.02], groups=2, first_group=1, n_atoms=3
    )
    starts_by_t = {300.0: starts, 500.0: starts}
    frames = [molecule("H2O")]
    trjs_by_t = {300.0: [frames, frames], 500.0: [frames, frames]}
    tasks = collect_sweep_replicate_tasks(
        grid, starts_by_t, trjs_by_t, settings, tmp_path, "sys"
    )
    assert len(tasks) == 8
    seeds = {t[1].base_seed for t in tasks}
    assert 42 + 10000 * 1 in seeds
    assert 42 + 10000 * 2 in seeds


def create_fake_replicate_dir(
    parent: Path,
    index: int,
    n_frames: int = 5,
) -> Path:
    """
    Create a fake replicate directory containing an xtb.trj file.

    :param parent: Base directory where rep_<index> is created.
    :type parent: Path
    :param index: Replicate numerical index.
    :type index: int
    :param n_frames: Number of atomic frames in trajectory.
    :type n_frames: int
    :return: Path to created trajectory file.
    :rtype: Path
    """
    rep_dir = parent / f"rep_{index}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    trj_path = rep_dir / "xtb.trj"
    frames = [molecule("H2O") for _ in range(n_frames)]
    ase.io.write(str(trj_path), frames, format="xyz")
    return trj_path


def test_resume_skips_existing(tmp_path: Path) -> None:
    """
    Verify groups with complete trajectories are skipped during task creation.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    grid = expand_sweep_grid(
        [300.0], [0.02], groups=2, first_group=1, n_atoms=3
    )
    starts = [("s0", molecule("H2O"))]
    settings = create_test_md_settings(replicates=1)
    g1_dir = sweep_group_dir(tmp_path, "sys", grid[0])
    create_fake_replicate_dir(g1_dir, 0)
    assert is_group_complete(g1_dir, 1)
    starts_by_t = {300.0: starts}
    trjs_by_t = {300.0: [[molecule("H2O")]]}
    tasks = collect_sweep_replicate_tasks(
        grid, starts_by_t, trjs_by_t, settings, tmp_path, "sys"
    )
    assert len(tasks) == 1
    assert tasks[0][1].base_seed == settings.base_seed + 20000


def create_mock_candidate_set(
    stats: dict[str, int] | None = None,
) -> CandidateSet:
    """
    Construct a mock CandidateSet instance with specified summary stats.

    :param stats: Optional dictionary of stats overrides.
    :type stats: dict[str, int] or None
    :return: Configured CandidateSet instance.
    :rtype: CandidateSet
    """
    st = {
        "events": 4,
        "reactive_replicates": 2,
        "rejected_multiple_radicals": 1,
    }
    if stats:
        st.update(stats)
    sp = Species("sp0", molecule("H2O"), "O", "XLYOFNOQVPJJNP", 1)
    cand = Candidate(
        "R001", ("sp0",), ("sp0",), ("sp0",), ("sp0",), (), "O>>O", 4, 0.0
    )
    return CandidateSet(("sp0",), {"sp0": sp}, (cand,), st)


def test_metrics_calculation_on_fake_freeze_output() -> None:
    """
    Verify core simulation and candidate metrics calculation formulas.

    :return: None.
    :rtype: None
    """
    cset = create_mock_candidate_set()
    setting = SweepSetting(300.0, 0.02, 0.06, 1)
    metrics = build_base_metrics(setting, 3, 5, cset)
    assert metrics["T"] == 300.0
    assert metrics["kpush_per_atom"] == 0.02
    assert metrics["group"] == 1
    assert metrics["n_atoms"] == 3
    assert metrics["kpush_eh"] == 0.06
    assert metrics["replicates"] == 5
    assert metrics["reactive_replicates"] == 2
    assert metrics["reactive_fraction"] == 0.4
    assert metrics["xtb_failed_replicates"] == 0
    assert metrics["events"] == 4
    assert metrics["rejected_multiple_radicals"] == 1
    assert metrics["radical_fraction"] == 0.25


def create_mock_candidate_set_with_reactions(
    reactions: Sequence[str],
) -> CandidateSet:
    """
    Create a CandidateSet containing candidates with specific reaction SMILES.

    :param reactions: Sequence of mapped reaction SMILES strings.
    :type reactions: Sequence[str]
    :return: CandidateSet with populated candidates.
    :rtype: CandidateSet
    """
    cands = []
    for i, rxn in enumerate(reactions):
        left, right = (tuple(side.split(".")) for side in rxn.split(">>"))
        mapped = f"[{i}]{rxn}"
        cands.append(
            Candidate(f"R{i:03d}", left, right, left, right, (), mapped, 1, 0.0)
        )
    return CandidateSet(("A",), {}, tuple(cands))


def test_compute_group_overlap() -> None:
    """
    Verify group overlap computes Jaccard similarity across group reactions.

    :return: None.
    :rtype: None
    """
    cset_1 = create_mock_candidate_set_with_reactions(["A>>B", "B>>C"])
    cset_2 = create_mock_candidate_set_with_reactions(["B>>C", "C>>D"])
    overlap = compute_group_overlap({1: cset_1, 2: cset_2})
    assert overlap is not None
    assert abs(overlap - (1.0 / 3.0)) < 1e-6
    # Same states, other atom numbers and the reverse direction still match.
    cset_3 = create_mock_candidate_set_with_reactions(["B>>A", "C>>B"])
    assert compute_group_overlap({1: cset_1, 3: cset_3}) == 1.0
    assert compute_group_overlap({1: cset_1}) is None


def create_candidate_set_with_smiles(smiles: str) -> CandidateSet:
    """
    Create a CandidateSet holding a single species with given SMILES.

    :param smiles: Target molecular SMILES string.
    :type smiles: str
    :return: CandidateSet containing the target species.
    :rtype: CandidateSet
    """
    mol = Chem.MolFromSmiles(smiles)
    inchikey = Chem.MolToInchiKey(mol) if mol else ""
    sp = Species("sp1", molecule("H2O"), smiles, inchikey, 1)
    cand = Candidate(
        "R001",
        ("sp1",),
        ("sp1",),
        ("sp1",),
        ("sp1",),
        (),
        f"{smiles}>>{smiles}",
        1,
        0.0,
    )
    return CandidateSet(("sp1",), {"sp1": sp}, (cand,))


def test_literature_match_aminomethanol() -> None:
    """
    Verify detection of aminomethanol from SMILES NCO against species InChIKey.

    :return: None.
    :rtype: None
    """
    lit = LiteratureProduct("aminomethanol", ("NCO",))
    matching_cset = create_candidate_set_with_smiles("NCO")
    non_matching = create_candidate_set_with_smiles("CC")
    assert candidate_set_holds_product(matching_cset, lit) is True
    assert candidate_set_holds_product(non_matching, lit) is False


def test_load_literature_products_from_file() -> None:
    """
    Verify parsing of literature.yaml reference specifications.

    :return: None.
    :rtype: None
    """
    products = load_literature_products("systems/hcho_nh3/literature.yaml")
    names = {p.name for p in products}
    assert "aminomethanol" in names
    assert "methanimine_water" in names
    amino = next(p for p in products if p.name == "aminomethanol")
    assert amino.smiles == ("NCO",)


def test_csv_row_replacement_on_rerun(tmp_path: Path) -> None:
    """
    Verify CSV row replacement on rerun indexed by (T, kpush_per_atom, group).

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    csv_file = tmp_path / "md_sweep.csv"
    initial = [{"T": 300.0, "kpush_per_atom": 0.015, "group": 1, "events": 5}]
    update_sweep_csv(csv_file, initial)
    rerun = [
        {"T": 300.0, "kpush_per_atom": 0.015, "group": 1, "events": 12},
        {"T": 300.0, "kpush_per_atom": 0.02, "group": 1, "events": 8},
    ]
    update_sweep_csv(csv_file, rerun)
    rows = read_existing_csv_rows(csv_file)
    assert len(rows) == 2
    assert rows[0]["events"] == "12"
    assert rows[1]["events"] == "8"


def test_merge_sweep_csv_rows_logic() -> None:
    """
    Verify in-memory row dictionary merging and replacement semantics.

    :return: None.
    :rtype: None
    """
    r1 = {"T": 300.0, "kpush_per_atom": 0.015, "group": 1, "val": 1}
    r2 = {"T": 300.0, "kpush_per_atom": 0.015, "group": 1, "val": 2}
    r3 = {"T": 600.0, "kpush_per_atom": 0.015, "group": 1, "val": 3}
    merged = merge_sweep_csv_rows([r1], [r2, r3])
    assert len(merged) == 2
    assert merged[0]["val"] == 2
    assert merged[1]["val"] == 3


def test_update_sweep_candidates_json(tmp_path: Path) -> None:
    """
    Verify candidate reaction labels dictionary updates and file writing.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    json_path = tmp_path / "md_sweep_candidates.json"
    update_sweep_candidates_json(json_path, {"T300_k0.02_g1": ["A>>B"]})
    update_sweep_candidates_json(json_path, {"T300_k0.02_g1": ["C>>D"]})
    import json

    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    assert data["T300_k0.02_g1"] == ["C>>D"]


def test_freeze_from_runs_union_no_collisions(tmp_path: Path) -> None:
    """
    Verify freeze --from-runs generates unique (folder, rep) identities.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    create_fake_replicate_dir(dir_a, 0)
    create_fake_replicate_dir(dir_b, 0)
    trjs, ids = load_from_runs_trajectories([dir_a, dir_b])
    out_dir = tmp_path / "cands"
    settings = create_test_md_settings()
    cset = freeze_system_candidates(
        molecule("H2O"), settings, trjs, "configs/engines.yaml", out_dir, ids
    )
    assert ids == [(str(dir_a), 0), (str(dir_b), 0)]
    assert (out_dir / "manifest.json").is_file()
    assert isinstance(cset, CandidateSet)


def create_test_replicate_trajectory(
    rep_dir: Path,
    energies: Sequence[str],
) -> Path:
    """
    Create a replicate directory with an annotated xtb.trj trajectory file.

    :param rep_dir: Target replicate directory path.
    :type rep_dir: Path
    :param energies: Sequence of energy values in Hartree.
    :type energies: Sequence[str]
    :return: Path to the generated trajectory file.
    :rtype: Path
    """
    rep_dir.mkdir(parents=True, exist_ok=True)
    comments = [
        f" energy: {e} gnorm: 0.1 xtb: 6.7.1 (unknown)" for e in energies
    ]
    atoms = "H 0.0 0.0 0.0\nH 0.0 0.0 1.0\nO 0.0 1.0 0.0\n"
    trj_path = rep_dir / "xtb.trj"
    trj_path.write_text("".join(f"3\n{c}\n{atoms}" for c in comments))
    return trj_path


def test_count_failed_replicates(tmp_path: Path) -> None:
    """
    Verify count_failed_replicates tallies only replicates with zero energy.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    group_dir = tmp_path / "group"
    create_test_replicate_trajectory(
        group_dir / "rep_0", ("-11.5", "-11.4", "0.000000000000", "-11.3")
    )
    create_test_replicate_trajectory(
        group_dir / "rep_1", ("-11.5", "-11.4", "-11.2", "-11.3")
    )
    assert count_failed_replicates(group_dir) == 1


def test_build_sweep_row_carries_failed_replicates() -> None:
    """
    Verify build_sweep_row records xtb_failed_replicates in output metrics.

    :return: None.
    :rtype: None
    """
    cset = create_mock_candidate_set()
    setting = SweepSetting(300.0, 0.02, 0.06, 1)
    row = build_sweep_row(
        setting, 3, 5, cset, (), 10.0, xtb_failed_replicates=2
    )
    assert row["xtb_failed_replicates"] == 2
    keys = list(row.keys())
    react_idx = keys.index("reactive_fraction")
    assert keys[react_idx + 1] == "xtb_failed_replicates"


def test_build_sweep_records_includes_failed_counts() -> None:
    """
    Verify build_sweep_records propagates failed replicate counts to rows.

    :return: None.
    :rtype: None
    """
    cset = create_mock_candidate_set()
    setting = SweepSetting(300.0, 0.02, 0.06, 1)
    rows, _ = build_sweep_records(
        [setting], 3, 5, {setting: cset}, (), 10.0, {}, {setting: 1}
    )
    assert rows[0]["xtb_failed_replicates"] == 1
    keys = list(rows[0].keys())
    react_idx = keys.index("reactive_fraction")
    assert keys[react_idx + 1] == "xtb_failed_replicates"
