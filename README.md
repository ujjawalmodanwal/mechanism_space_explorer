# Mechanism Space Explorer

Nanoreactor reaction networks with machine-learned potentials.

**Read the report:** [reports/present_study/main.pdf](reports/present_study/main.pdf)

If we swap DFT for a machine-learned potential, does a nanoreactor predict
different products, or a different route to them? This repository answers
that for small C/H/N/O systems. Every engine refines the same frozen set
of candidate reactions, so any difference between the networks comes from
the engine alone.

Engines: GFN2-xTB, MACE-OMol25, UMA-S 1.2.1, AIMNet2 and ANI-1xnr. DFT
(ORCA, ωB97M-V/def2-TZVPD) checks them with single points on their own
structures. Exact checkpoints are under "Models and licenses".

Status: all five engines have refined all four systems. DFT single points
have checked every stored saddle and molecule (467 structures) and 80
reaction paths (`PLAN.md`, D23). The network comparison has not run yet.
Results, limits and next steps: the report above, and
[REPORT.md](REPORT.md) in plain text.

## How it works

1. **Discover.** Hot xTB molecular dynamics with a soft spherical wall and
   metadynamics pushes the molecules to react, in many short replicates
   (`nrnet discover`).
2. **Freeze.** Every lasting change of bonds is an event. Events are
   grouped by the molecules that react, both sides are relaxed with xTB,
   and up to three example geometries are kept per reaction. The result is
   written to `candidates/<system>/` with a sha256 manifest and committed
   (`nrnet freeze`).
3. **Refine.** Each engine relaxes every species, checks it is a minimum,
   and computes its free energy (quasi-harmonic, 1 atm). For each reaction
   it searches a transition state: either its own band (aligned endpoints,
   a plain NEB, then a climbing NEB) or, with `--seed-from`, another
   engine's transition state as the start. Either way the engine polishes
   the saddle with Sella, requires exactly one imaginary mode, and follows
   that mode downhill to confirm both ends (`nrnet refine`).
4. **Rank.** A network node is the state of the whole box. Each confirmed
   reaction adds two arrows with Eyring rates. Five rankers then predict
   the products, from plain thermodynamics (R0) to a kinetic simulation
   (R4) (`nrnet rank`).
5. **Evaluate.** Each engine is compared with the reference: barriers and
   reaction energies (E1), predicted products (E2), how well cheap rankers
   match the kinetic one (E3), and cost (E4) (`nrnet evaluate`). E5, how
   step errors carry into the predicted products, is planned (`PLAN.md`).
6. **Figures.** Four figures and three tables are drawn from those files
   (`nrnet figures`).

`PLAN.md` holds the design decisions and every finding from the runs.

## Systems

| ID | Reactants in the box | Products to check for | Reference values | Ranked at (D13) |
|---|---|---|---|---|
| `hcho_nh3` | HCHO + NH3 | aminomethanol (NH2CH2OH), then methanimine (CH2=NH) + H2O | addition barrier ≈ 34 kcal/mol; aminomethanol dehydration barrier 230–234 kJ/mol | 298.15 K, 600 K |
| `hcooh` | HCOOH; replicates alternate between the trans and cis conformers | CO + H2O (dehydration), CO2 + H2 (decarboxylation) | gas phase: dehydration dominant, despite a similar decarboxylation barrier | 1000 K |
| `hcooh_h2o` (optional) | HCOOH + H2O | the same two channels | with water, decarboxylation becomes favored | 1000 K |
| `cpd_ma` | cyclopentadiene + maleic anhydride; replicates alternate between endo and exo approaches | endo and exo norbornene-2,3-dicarboxylic anhydride (stereo-aware identity, D12) | endo kinetic, exo thermodynamic; ΔΔG‡ = 2.2 kcal/mol favoring endo; 45:55 endo/exo after heating at 220 °C | 298.15 K, 493.15 K |

Formic acid is ranked at 1000 K only: its gas-phase decomposition
barriers are high, so at room temperature nothing reacts within the
one-hour observation time. Each system's inputs are in
`systems/<id>/`. Sources for the reference values: to add.

## Reproducibility

- **How the candidates were found** (MD settings, the checks run on
  the results, the faults they exposed and each fix, with numbers) is
  recorded in [docs/discovery-log.md](docs/discovery-log.md). The
  check scripts are in `scripts/checks/`.
- **Molecular dynamics reproduces in distribution only.** The seed fixes
  the start exactly, but xtb adds about 1e-7 Å of run-to-run noise per
  frame, and hot dynamics grows it to whole ångströms within a
  picosecond. Rerunning `nrnet discover` gives a similar candidate set,
  not the same one. That is why the frozen candidates are committed: they
  are the fixed input of everything after them.
- **Every model is pinned** to one checkpoint in `configs/engines.yaml`,
  and every library to one version in `uv.lock`.
- Each engine call is logged to `results/refine/<system>/<engine>/calls.csv`
  with wall time, force calls, the git commit and a hash of the configs.

## Setup

The core environment (xTB, freezing, ranking, evaluation, figures):

```bash
uv sync --extra xtb
```

Each ML engine has its own environment, because their dependencies
conflict:

```bash
UV_PROJECT_ENVIRONMENT=.venvs/mace uv sync --extra mace
UV_PROJECT_ENVIRONMENT=.venvs/uma uv sync --extra uma
UV_PROJECT_ENVIRONMENT=.venvs/aimnet uv sync --extra aimnet
UV_PROJECT_ENVIRONMENT=.venvs/ani uv sync --extra ani
```

- torchani downloads the ANI-1xnr weights on first use.
- UMA needs a Hugging Face account that has accepted the model's terms,
  then `hf auth login`.
- DFT needs ORCA. Set `NRNET_ORCA` to the absolute path of the `orca`
  executable; the run settings are in `configs/engines.yaml`. Parallel
  runs need the OpenMPI version ORCA was built with (4.1.x for ORCA
  6.1.1) first on `PATH`. On macOS, `DYLD_LIBRARY_PATH` is dropped when
  ORCA starts `mpirun` through the shell, so put a small `mpirun` wrapper
  first on `PATH` that sets it and calls the real `mpirun` with
  `-x DYLD_LIBRARY_PATH`.

## Running system 1

The frozen candidates are committed, so a fresh clone can start at
`refine`. `discover` and `freeze` are shown for completeness.

```bash
uv run nrnet discover --system hcho_nh3
uv run nrnet freeze --system hcho_nh3
uv run nrnet refine --system hcho_nh3 --engine xtb
.venvs/aimnet/bin/nrnet refine --system hcho_nh3 --engine aimnet \
    --seed-from xtb
.venvs/mace/bin/nrnet refine --system hcho_nh3 --engine mace \
    --seed-from xtb
.venvs/uma/bin/nrnet refine --system hcho_nh3 --engine uma \
    --seed-from xtb
.venvs/ani/bin/nrnet refine --system hcho_nh3 --engine ani1xnr \
    --seed-from xtb
```

DFT does not refine here (`PLAN.md`, D23): it gives single points on the
engines' stored structures and along their IRC paths. The scripts and
their commands are in [scripts/checks/README.md](scripts/checks/README.md)
(`dft_single_points.py`, `ml_irc.py`, `dft_path_points.py`,
`dft_compare.py`).

Then rank, evaluate and draw, against one engine as the reference:

```bash
make figures SYSTEM=hcho_nh3 REFERENCE=xtb \
    COMPARE=mace,uma,aimnet,ani1xnr
```

The report's figures and tables come from
`scripts/report/report_figures.py`. `refine` resumes: finished species and
reactions are skipped on rerun.

## Known limits

- The kinetics describe one copy of the box, so a two-molecule step has
  a fixed 1 atm rate.
- A step with no barrier (a bond simply breaking into two radicals) has
  no saddle. It keeps its arrow with G(TS) = max(G_A, G_B), one rule for
  every engine (`PLAN.md`, D11), which is a lower bound on the true
  free-energy barrier.
- An event that is really two steps is reported as `endpoint_mismatch`,
  not split.
- ANI-1xnr takes no charge or spin, so it runs on closed-shell molecules
  only; a radical gives `unsupported`, and its network lacks radical
  steps (9 of the 65 reactions). It was trained at BLYP/TZV2P, a lower
  level than the OMol25 models, so it differs from them in data and level
  at once.
- A species is identified by its bonds only, so stereoisomers (for
  example endo and exo Diels–Alder adducts) share one identity. This must
  change before a system with stereoisomers is run (`PLAN.md`, D12).
- The rotational symmetry number is 1 for every molecule, the same error
  for every engine.

## Models and licenses

The MIT license in `LICENSE` covers this repository's code only. No model
weights are committed; each comes from its source under its own terms.

| Engine | Checkpoint | Source | Weights license |
|---|---|---|---|
| GFN2-xTB | method, no weights | tblite 0.8.0; xtb 6.7.1 for MD | LGPL-3.0 |
| MACE-OMol25 | `MACE-omol-0-extra-large-1024.model` | ACEsuit mace-foundations, release `mace_omol_0` | Academic Software License (non-commercial) |
| UMA | `uma-s-1p2p1.pt`, task `omol`, MD5 `3497615f…` | facebook/UMA on Hugging Face (gated) | FAIR Chemistry License v1 and its Acceptable Use Policy; publications must acknowledge use |
| AIMNet2 | `aimnet2-nse_0` (spin-aware NSE), sha256 `b98b3449…` | aimnet 0.2.6 model registry | MIT (NSE model card) |
| ANI-1xnr | `ani1xnr.pt` (8-network ensemble), sha256 `beef5418…` | torchani 2.12 `models.ANI1xnr`, from atomistic-ml/ani-1xnr | No license stated in the weights repository; torchani code is MIT; not redistributed here |
| DFT | ωB97M-V/def2-TZVPD | ORCA, version recorded per run | ORCA end-user license (free for academic use, not redistributed) |

## Citing

If you use this work, please also cite the methods and models it builds
on:

- **Nanoreactor.** L.-P. Wang, A. Titov, R. McGibbon, F. Liu,
  V. S. Pande, T. J. Martínez, "Discovering chemistry with an ab initio
  nanoreactor", *Nat. Chem.* 6, 1044 (2014).
- **Metadynamics reaction exploration with xTB.** S. Grimme,
  "Exploration of chemical compound, conformer, and reaction space with
  meta-dynamics simulations based on tight-binding quantum chemical
  calculations", *J. Chem. Theory Comput.* 15, 2847 (2019).
- **GFN2-xTB.** C. Bannwarth, S. Ehlert, S. Grimme, *J. Chem. Theory
  Comput.* 15, 1652 (2019).
- **ORCA.** F. Neese, "Software update: The ORCA program system—Version
  6.0", *WIREs Comput. Mol. Sci.* 15, e70019 (2025).
- **ωB97M-V.** N. Mardirossian, M. Head-Gordon, *J. Chem. Phys.* 144,
  214110 (2016).
- **OMol25.** D. S. Levine et al., "The Open Molecules 2025 (OMol25)
  dataset, evaluations, and models", arXiv:2505.08762 (2025).
- **UMA.** B. M. Wood et al., "UMA: A family of universal models for
  atoms", arXiv:2506.23971v2 (2026).
- **MACE.** I. Batatia, D. P. Kovács, G. N. C. Simm, C. Ortner,
  G. Csányi, "MACE: Higher order equivariant message passing neural
  networks for fast and accurate force fields", NeurIPS (2022).
  MACE-OMol25 asks for the OMol25 paper as well.
- **AIMNet2.** D. M. Anstine, R. Zubatyuk, O. Isayev, "AIMNet2: a neural
  network potential to meet your neutral, charged, organic, and
  elemental-organic needs", *Chem. Sci.* (2025).
- **ANI-1xnr.** S. Zhang et al., "Exploring the frontiers of
  condensed-phase chemistry with a general reactive machine learning
  potential", *Nat. Chem.* (2024), doi:10.1038/s41557-023-01427-3.
- **TorchANI.** X. Gao, F. Ramezanghorbani, O. Isayev, J. S. Smith,
  A. E. Roitberg, "TorchANI: A free and open source PyTorch-based deep
  learning implementation of the ANI neural network potentials",
  *J. Chem. Inf. Model.* 60, 3408 (2020).
- **Sella.** E. D. Hermes, K. Sargsyan, H. N. Najm, J. Zádor, "Sella, an
  open-source automation-friendly molecular saddle point optimizer",
  *J. Chem. Theory Comput.* 18, 6974 (2022).
- **Climbing-image NEB.** G. Henkelman, B. P. Uberuaga, H. Jónsson,
  *J. Chem. Phys.* 113, 9901 (2000).
- **Quasi-harmonic correction.** R. F. Ribeiro, A. V. Marenich,
  C. J. Cramer, D. G. Truhlar, *J. Phys. Chem. B* 115, 14556 (2011).
- **ASE.** A. H. Larsen et al., *J. Phys.: Condens. Matter* 29, 273002
  (2017).

## License

MIT for the code, see `LICENSE`. Model weights carry their own licenses,
listed above.
