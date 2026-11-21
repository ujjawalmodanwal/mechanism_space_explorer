# Simulation Checks and Diagnostics

Reproducible diagnostic scripts for nanoreactor trajectory analysis, covalent
bond audits, replicate convergence comparisons, and candidate set inspection.
Each script writes structured CSV output under `data/<system>/checks/` for
downstream plotting and quality control.

## census.py

Audits all chemical species produced during nanoreactor MD trajectories.
Samples frames at 1 ps intervals (plus the terminal frame) without event-based
filtering, relaxes each full simulation box with GFN2-xTB (via tblite BFGS),
fragments the resulting topology, and tests whether each observed molecule
exists within the frozen candidate sets of the system.
Writes `data/<system>/checks/census.csv` with columns `key`, `formula`,
`smiles`, `inchikey14`, `runs_with`, `in_frozen_sets`, `first_run`, and
`first_frame`.

```bash
PYTHONPATH=src .venv/bin/python scripts/checks/census.py \
  --system cpd_ma \
  --runs "runs/sweep/cpd_ma/T300_k*/g1" \
  --workers 10
```

## bond_audit.py

Evaluates the sensitivity of reaction event detection across covalent radius
scale cutoffs (1.1, 1.2, and 1.3) on unrelaxed trajectory frames. Extracts
stable state transitions using `find_events` with settling frames, flags
transitions generating multiple radical centers, and tallies kept versus
rejected events.
Writes `data/<system>/checks/bond_audit.csv` with columns `scale`,
`state_before`, `state_after`, `multiple_radicals`, and `count`.

```bash
PYTHONPATH=src .venv/bin/python scripts/checks/bond_audit.py \
  --system hcho_nh3 \
  --runs "runs/sweep/hcho_nh3/T300_k*/g*" \
  --workers 4
```

## compare_groups.py

Compares reaction discovery overlap and event counts between main replicate
groups (e.g. g1-4) and check replicate groups (e.g. g5-6) across metadynamics
push constants (Phase B consistency check). Summarizes shared reaction
coverage and prints individual reaction distributions.
Writes `data/<system>/checks/group_comparison.csv` with columns `reaction`,
`main_events`, and `check_events`.

```bash
PYTHONPATH=src .venv/bin/python scripts/checks/compare_groups.py \
  --system hcooh \
  --kpush 0.03,0.035 \
  --main 1-4 \
  --check 5-6
```

## list_reactions.py

Inspects a frozen candidate directory, displaying each reaction's identifier,
event count, endpoint pair count, simulation source, literature reference, and
reactant and product species expressed as SMILES.
When invoked with `--csv PATH`, writes the reaction table to the specified
CSV file with columns `reaction_id`, `n_events`, `pairs`, `source`,
`literature_name`, `state_a`, and `state_b`.

```bash
PYTHONPATH=src .venv/bin/python scripts/checks/list_reactions.py \
  tests/data/hcho_nh3_candidates \
  --csv data/hcho_nh3/checks/reactions.csv
```

## refine_summary.py

Summarizes one engine's refinement of a frozen set: one row per reaction and
ranking temperature with the search status, route (band or seed), seed
engine, attempts, imaginary frequency, forward and reverse free-energy
barriers and dG_rxn in kcal/mol. Barriers use the network's rule, G_TS minus
the G of the reacting molecules only. Prints the species and reaction status
counts and writes `data/<system>/checks/refine_<engine>.csv`.

```bash
PYTHONPATH=src .venv/bin/python scripts/checks/refine_summary.py \
  --system hcho_nh3 \
  --engine xtb
```

## orca_benchmark.py

Times ORCA electronic structure calculations (single-point energy, gradient,
and vibrational frequencies) across structures and CPU process counts at the
study's reference DFT level (wB97M-V/def2-TZVPD). Evaluates wall-clock scaling
and per-core memory budgets to determine DFT refinement strategy.
Writes `data/benchmarks/orca_benchmark.csv` with columns `system`, `engine`,
`reaction_id`, `atoms`, `task`, `nprocs`, `maxcore_mb`, `wall_s`,
`orca_run_s`, `scf_cycles`, `energy_eh`, `n_imag`, `orca_version`, `status`,
and `note`.

```bash
PYTHONPATH=src .venv/bin/python scripts/checks/orca_benchmark.py \
  --structure hcooh:xtb:R001 \
  --task SP \
  --task Freq \
  --nprocs 4 \
  --nprocs 8
```

