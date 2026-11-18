# Nanoreactor networks with ML potentials

If we swap DFT for a machine-learned potential, does a nanoreactor predict
different products, or a different route to them? This repository answers
that for small C/H/N/O systems. Every engine refines the same frozen set
of candidate reactions, so any difference between the networks comes from
the engine alone.

Engines: GFN2-xTB, MACE-OMol25, UMA, AIMNet2, Egret-1, and DFT
(ORCA, wB97M-V/def2-TZVPD) as the reference.

Status: in progress. The pipeline runs end to end on system 1
(HCHO + NH3) with xTB, AIMNet2, MACE and Egret-1; DFT and UMA are next.

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
   match the kinetic one (E3), and cost (E4) (`nrnet evaluate`).
6. **Figures.** Four figures and three tables are drawn from those files
   (`nrnet figures`).

`PLAN.md` holds the design decisions and every finding from the runs.

## Reproducibility

- **Molecular dynamics reproduces in distribution only.** The seed fixes
  the start exactly, but xtb adds about 1e-7 Å of run-to-run noise per
  frame, and hot dynamics grows it to whole ångströms within a
  picosecond. Rerunning `nrnet discover` gives a similar candidate set,
  not the same one. That is why the frozen candidates are committed: they
  are the fixed input of everything after them.
- Each engine call is logged to `results/refine/<system>/<engine>/calls.csv`
  with wall time, force calls, the git commit and a hash of the configs.

## Setup

The core environment (xTB, freezing, ranking, evaluation, figures):

```bash
uv sync --extra xtb
```

Each ML engine has its own environment, because their dependencies
conflict. The `mace` environment also serves Egret-1:

```bash
UV_PROJECT_ENVIRONMENT=.venvs/mace uv sync --extra mace
UV_PROJECT_ENVIRONMENT=.venvs/uma uv sync --extra uma
UV_PROJECT_ENVIRONMENT=.venvs/aimnet uv sync --extra aimnet
```

- Egret-1 weights go in `models/` (not committed); see Rowan's Egret-1
  repository.
- UMA needs a Hugging Face account that has accepted the model's terms.
- DFT needs ORCA. Set `NRNET_ORCA` to the absolute path of the `orca`
  executable; the run settings are in `configs/engines.yaml`.

## Running system 1

```bash
uv run nrnet discover --system hcho_nh3
uv run nrnet freeze --system hcho_nh3
uv run nrnet refine --system hcho_nh3 --engine xtb
.venvs/aimnet/bin/nrnet refine --system hcho_nh3 --engine aimnet \
    --seed-from xtb
.venvs/mace/bin/nrnet refine --system hcho_nh3 --engine mace \
    --seed-from xtb
make figures SYSTEM=hcho_nh3 REFERENCE=dft COMPARE=xtb,mace,aimnet,egret
```

`refine` resumes: finished species and reactions are skipped on rerun.

## Known limits

- The kinetics describe one copy of the box, so a two-molecule step has
  a fixed 1 atm rate.
- A step with no barrier (a bond simply breaking into two radicals) has
  no saddle and is left out of the network for now.
- An event that is really two steps is reported as `endpoint_mismatch`,
  not split.
- Egret-1 handles closed-shell molecules only; a radical gives
  `unsupported`.

## License

MIT, see `LICENSE`.
