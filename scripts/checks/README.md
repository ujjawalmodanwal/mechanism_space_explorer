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

## force_call_benchmark.py

Times single force call evaluations (energy and gradients) for an engine
across molecular structures with coordinate jitter under a uniform protocol.
Records timing metrics to evaluate per-engine force call throughput.
Writes `data/benchmarks/force_call_benchmark.csv` with columns `timestamp_utc`,
`engine`, `model`, `structure`, `n_atoms`, `threads`, `warmup`, `calls`,
`jitter_angstrom`, `median_ms`, `mean_ms`, `min_ms`, and `max_ms`.

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python \
  scripts/checks/force_call_benchmark.py \
  --engine xtb \
  --structure systems/cpd_ma/reactant_endo.xyz \
  --structure cpd_ma:xtb:R001
```


## engine_pair.py

Compares two engines' kept saddles on every reaction both have confirmed.
Reads the two `refine_<engine>.csv` summaries, the reaction records and the
saddle geometries; a small saddle RMSD with a large barrier gap means the two
potentials disagree on the same saddle, not that the search found different
ones. The RMSD uses the stored atom order and tries no permutations.
Writes `data/<system>/checks/pair_<A>_<B>.csv` with columns `reaction_id`,
`temperature_k`, `literature_name`, `dg_act_a_kcal`, `dg_act_b_kcal`,
`dg_act_diff_kcal` (B minus A), `de_ts_rcx_diff_kcal`, `de_ts_pcx_diff_kcal`,
`imaginary_a_cm`, `imaginary_b_cm`, and `rmsd_a`.

```bash
PYTHONPATH=src .venvs/mace/bin/python scripts/checks/engine_pair.py \
  --system cpd_ma \
  --engines uma mace
```

## dft_single_points.py

DFT on every structure the engines stored (PLAN D23): EnGrad on each
confirmed saddle (energy and force), SP on each molecule, at the study's DFT
level, on the stored geometry, with no optimization. Two ORCA jobs at a time
by default (4 processes, 1500 MB per core each). Resumable: structures with
an ok row are skipped. Writes `results/dft_sp/<system>/<engine>/points.jsonl`
(one row per structure: `uid`, `kind`, `id`, `task`, `atoms`, `multiplicity`,
`ok`, `energy_eh`, `energy_ev`, `fmax_ev_a`, `frms_ev_a`, `forces_ev_a`,
`scf_cycles`, `wall_s`, `orca_run_s`, `orca_version`, `note`); ORCA files in
`runs/dft_sp/`.

```bash
NRNET_ORCA=$HOME/opt/orca_6_1_1/orca \
PATH=$HOME/opt/openmpi-4.1.8/wrap:$HOME/opt/openmpi-4.1.8/bin:$PATH \
PYTHONPATH=src .venv/bin/python scripts/checks/dft_single_points.py \
  --workers 2 --nprocs 4 --maxcore-mb 1500
```

## ml_irc.py

IRC on each engine's confirmed saddles with that engine (Sella, both sides,
dx 0.1, to 0.05 eV/A, at most 300 steps a side). When Sella's inner loop
fails, it reruns with `keep_going` and marks the row. Run with the engine's
own environment. Writes `results/irc/<system>/<engine>/<reaction_id>.extxyz`
(reverse end first, engine energy per frame) and `paths.jsonl` (`ok`,
`keep_going`, `inner_failures`, steps and convergence per side, `n_points`,
`ts_index`, `peak_index`, `ends_match`, `wall_s`).

```bash
PYTHONPATH=src .venvs/uma/bin/python scripts/checks/ml_irc.py \
  --system cpd_ma --engine uma --literature-only
```

## dft_path_points.py

DFT single points along the IRC paths: both ends and `--per-side` frames on
each side first, then up to `--fine-limit` frames around the highest DFT
point, repeated until its neighbours are done. The saddle frame's energy is
taken from `results/dft_sp`. The highest DFT energy on a path is an upper
limit to the DFT saddle energy. Writes
`results/dft_path/<system>/<engine>/points.jsonl` (uid
`path:<reaction_id>_<index>`).

```bash
PYTHONPATH=src .venv/bin/python scripts/checks/dft_path_points.py \
  --system cpd_ma --per-side 4 --fine-limit 4
```

## dft_compare.py

One row per confirmed reaction and engine: barrier, reverse barrier and
reaction energy (kcal/mol, from the separated molecules of each side) from
the engine and from DFT on the same geometries, their differences, the DFT
force on the saddle, free energy barriers at each ranking temperature (DFT
energy plus the engine's own G - E), and from the IRC path the DFT peak,
its height above the DFT energy at the saddle and its offset in frames.
Per reaction, `bound_best` (lowest path peak over the engines, an upper
limit) and `saddle_sp_best` (lowest DFT energy at an engine's saddle), both
from the lowest DFT energy of each reactant molecule over the engines.
Writes `data/<system>/checks/dft_compare.csv`.

```bash
PYTHONPATH=src .venv/bin/python scripts/checks/dft_compare.py
```

## dft_summary.py

Summarizes DFT comparison metrics across engines and systems: rows, count with
barrier error, mean absolute, mean signed, min and max barrier error, mean
absolute reaction error, median and max DFT fmax, rows with ok IRC path status,
and median and max peak height above the engine saddle.
Writes `data/<system>/checks/dft_summary.csv` and prints a markdown table per
system.

```bash
PYTHONPATH=src .venv/bin/python scripts/checks/dft_summary.py
```

## saddle_geometry.py

Inspects refined saddle geometries and DFT forces for a reaction across
engines: the N shortest interatomic distances (labelled like C0-H3), the norm
of the DFT force on each atom from single points, the DFT fmax, and the DFT
energy above the lowest DFT saddle energy over all engines in kcal/mol.
Prints one line per engine and writes
`data/<system>/checks/saddle_geometry_<reaction>.csv`.

```bash
PYTHONPATH=src .venv/bin/python scripts/checks/saddle_geometry.py \
  --system hcho_nh3 --reaction R006
```
