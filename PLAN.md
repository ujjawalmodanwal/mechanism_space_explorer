# Mechanism Space Explorer: implementation plan

Nanoreactor reaction networks with machine-learned potentials.

Draft of 2026-11-17. Sources: the one-week study plan (nanoreactor.pdf)
and afscripts' `scripts/xtb_nanoreactor.py` with its helpers.

## 1. What we are building

The study asks one question: if we swap DFT for an ML potential, does a
nanoreactor predict different products, or a different pathway to them?

To answer it fairly, the expensive, random part (hot MD that discovers
reactions) runs once, with xTB. Its output is frozen to disk: the
molecules it found and the reactions between them. Then each of six
energy engines (GFN2-xTB, MACE-OMol25, UMA, AIMNet2, ANI-1xnr and DFT)
takes the same frozen file, re-optimizes every molecule, searches
every transition state, and computes free energies. Because the input is
identical, any difference between the networks comes from the engine
alone.

Each engine's network is then ranked five ways (R0 to R4, from a plain
free-energy filter to a kinetic simulation), and four experiments (E1 to
E4) compare engines step by step, network by network, ranker by ranker,
and by cost.

## 2. What the existing script does, and what changes

Today's script runs N replicates of xTB MD with metadynamics through the
`xtb` binary, in a process pool. For every frame it perceives bonds with
RDKit and writes one canonical SMILES for the whole system. When the
SMILES changes, it adds an edge to a networkx graph. Replicate graphs are
merged, nodes can be relaxed with SevenNet, and NEB plus Sella runs
between node geometries to get an activation energy in eV.

Five things stop it from answering the study's question:

1. **No confinement.** The xTB input has no `$wall`, so HCHO and NH3
   drift apart instead of colliding. It is not a nanoreactor yet. Fix: a
   logfermi spherical wall around all atoms.
2. **Bond flicker.** A bond near the cutoff switches on and off from one
   frame to the next, and every switch becomes an "edge". Fix: a new
   state counts only if it lasts a minimum time (the lifetime filter).
3. **Bond orders from hot frames.** RDKit has to guess bond orders from
   distorted, 2000 K geometries and radicals. When it fails, the frame is
   skipped without a word. Fix: track states by connectivity (who is
   bonded to whom), which is what MD actually tells us. Bond orders,
   SMILES and InChIKeys are computed later, on relaxed geometries, as
   labels.
4. **TS endpoints from unrelated moments.** The NEB starts from node
   geometries that may come from different events or replicates, so the
   two ends are not in a reacting pose. Fix: keep, for each event, the
   frame just before and just after, from the same replicate. Same
   atoms, same order, same moment. This also gives the atom mapping for
   free.
5. **Missing pieces.** Endpoints are not optimized before the NEB.
   Nothing checks for exactly one imaginary frequency or confirms the TS
   joins the right ends. There are no free energies, reverse barriers,
   atom-mapped reactions or per-call timing.

What we drop: the SevenNet refiner and its registry, registries with a
single entry (the reaction filter, the TS searcher), the GSM stub, the
three sort flags (the rankers replace them), the SDF export, and the
graph merge. Replicates will return lists of events, so merging becomes
joining lists.

What we keep: the replicate pool, the xTB MD and metadynamics input, and
the NEB-then-Sella idea.

The new repository is public. Rather than copy afscripts files, we
rewrite the parts we need into the modules below. That is both the
simplification and the cleaner answer to the "is this your own code"
check. Publishing still needs your yes (section 10).

## 3. Decisions, including where this departs from the PDF

- **D1. A node is the state of the whole system.** The nanoreactor holds
  every atom in one box, so a state is "these molecules, together", for
  example `{CH2O, NH3}` or `{CH5NO}`. Every reaction is then a plain
  arrow between two states. Dijkstra, branching probabilities and the
  rate equations all work unchanged. The PDF keys nodes by species
  (InChIKey plus charge and spin). With species as nodes, HCHO + NH3 ->
  aminomethanol needs an arrow with two tails (a hyperedge), and none of
  the rankers handle that. We keep the species key inside the state:
  a state's key is the sorted list of its species' keys.
- **D2. A species key is connectivity, not bond orders.** It is the
  formula plus a hash of the bonded graph, for example `CH5NO-3f9a1c`.
  InChIKey and SMILES are stored beside it when RDKit can perceive them.
- **D3. Barriers are measured from separated molecules.** Only the
  molecules that react are computed in a TS search. Spectator molecules
  cancel, because they sit on both sides. The barrier is
  G(TS) minus the sum of G(each reacting molecule alone). Standard state
  is 1 atm for every engine. The 1 M correction is left out, and the
  same for all engines.
- **D4. Configs.** One `systems/<name>/system.yaml` (MD settings and the
  seed) and one `configs/engines.yaml`, not one file per engine and
  system (15 near-copies). Each result row records the system, engine,
  a hash of both configs, and the git commit.
- **D5. Environments.** uv with `pyproject.toml` extras, not conda (none
  on the Mac). MACE and UMA have pinned different versions of a shared
  library (e3nn) before, so each engine gets its own environment. That
  costs nothing here: each engine already runs as its own command.
  ANI-1xnr has its own environment (uv extra `ani`).
- **D6. Where things run.** The molecules are small (under ~30 atoms),
  so xTB and all three ML engines run on the Mac's CPU. Only DFT goes to
  the cluster (ORCA). The PDF assumed a GPU; we don't need one.
- **D7. Kinetic yields are read at a set time.** Every reaction is
  reversible, so if we wait forever, the kinetic simulation always ends
  at the Boltzmann distribution, which is the free-energy ranking again.
  R4 reports yields at an observation time `t_obs` (default 1 hour),
  plus the whole curve.
- **D8. Low modes.** Real frequencies below 100 cm^-1 are raised to
  100 cm^-1 (Truhlar's quasi-harmonic fix) before thermochemistry. The
  rotational symmetry number is 1 for every molecule. That is a small
  error (RT ln 3 = 0.65 kcal/mol for NH3 at 298 K), but it is identical
  across engines, so the comparison is unaffected. It is written down
  as a limit.
- **D9. Spin.** All systems are neutral. Multiplicity comes from the
  electron count: even means singlet, odd means doublet. Triplets and
  open-shell singlets are out of scope and listed as a limit. ANI-1xnr
  supports only neutral closed-shell molecules (it takes no charge or
  spin, settings `closed_shell_only`). A doublet given to ANI-1xnr gets
  status `unsupported` without a call, and its reactions are missing from
  ANI-1xnr's network. E2 reports that as a coverage gap, not an accuracy
  error.
- **D10. Code style.** The AtomField code style:
  snake_case, inline type hints, Sphinx docstrings, 80 characters per
  line, methods of at most 15 lines, no underscore functions. It reads
  like ordinary Python to an outside reader.
- **D11. A barrierless step keeps its arrow, through the clamp rule**
  (decided 2026-11-18). Some steps have no saddle on an engine's
  surface: a bond breaks into two fragments, the band's highest image
  sits at an end, and Sella finds no single imaginary mode. Dropping
  them can delete real channels at high temperature. Such a step gets
  G(TS) = max(G_A, G_B), with both sides measured as in D3 (separated
  molecules, 1 atm). This is the rule that already clamps a confirmed
  TS lying below an endpoint, `max(G_raw_TS, G_A, G_B)` in
  `network.apply_barrier`, with the raw TS term absent. So the uphill
  direction's barrier is dG_rxn and the downhill one is zero.
  - Detection: after the climbing stage, no interior image rises more
    than 0.04 eV (about 1 kcal/mol, a `search.yaml` setting) above the
    higher endpoint, and both ends still relax to the expected states.
    The status is `barrierless`, not a failure.
  - These arrows carry `barrierless: true`. R4 runs twice, with and
    without them. If the predicted products change, that is a result
    and gets reported.
  - Known bias: the products' full translational entropy at 1 atm makes
    dissociation too fast at high temperature, because a real loose TS
    has less entropy than fully separated fragments. The bias is the
    same for every engine, so the comparison stays fair. It is listed
    in the README's Known limits.
- **D12. Species identity becomes stereo-aware before system 3.**
  Connectivity keys (D2) merge stereoisomers. Endo and exo Diels–Alder
  adducts have one bond graph, and the figure bug where two CH3NO
  isomers shared one label showed that identity decides what the tables
  say. Before system 3 runs, a species key becomes formula,
  connectivity hash and a stereo descriptor. The descriptor comes from
  the relaxed 3D geometry: the parity of each stereocentre (signed
  volume of its neighbours, ordered by the Weisfeiler-Lehman labels)
  and E/Z for each stereogenic double bond. RDKit's InChIKey stays a
  label and a cross-check. MD events stay connectivity-only, since hot
  frames have no stable stereo. The descriptor is assigned at freeze,
  on the xTB-relaxed ends, and every engine's endpoint check compares
  the full key. Endo and exo then become two candidates with their own
  TS searches. An achiral species gets an empty descriptor, so frozen
  sets without stereocentres keep their keys (a test checks this).
- **D13. Ranking temperatures are set per system.** Each
  `systems/<name>/system.yaml` names its own ranking temperatures (the
  temperature of interest from its literature reference, plus one
  elevated temperature), replacing the single list in `search.yaml`.
  The MD temperature stays a sampling choice only. The README's Systems
  table lists them.
- **D14. Engines are also compared on what all of them can compute.**
  ANI-1xnr cannot treat radicals (D9), so its network lacks exactly
  the radical and barrierless steps. E1 and E2 therefore report every
  engine twice: on its own network, and on the common subset (reactions
  every engine could compute, states every engine could reach).
  Differences on the common subset measure accuracy; the rest measures
  coverage.
- **D15. Discovery settings come from the literature, then a sweep**
  (2026-11-18). Zhang, Xu, Lan (JCTC 2023) ran the same HCHO + NH3
  nanoreactor: bath 300 K, kpush 0.20 and 0.25 Eh, alpha 0.7 bohr^-2,
  one reference structure replaced every 1 ps, a 4.3 A logfermi wall
  (xtb defaults, 6000 K and beta 10 bohr^-1), 5-10 ps runs, starts taken
  from a 500 ps unbiased MD inside the wall. Their sweep found 0%, 0%,
  12% and 76% reactive runs at kpush 0.10, 0.15, 0.20 and 0.25. Grimme
  (JCTC 2019) uses kpush about 0.02 x N atoms, alpha 0.6 and a 298 K
  bath. The pattern: a cool bath, kpush about 0.02-0.03 Eh per atom
  with a narrow window (none below, fragmentation above), alpha
  0.6-0.7, the wall sized by density, frames under 10 fs apart, and a
  second independent group of runs to check convergence.
  - Defaults in every `system.yaml` follow it; the wall is Zhang's
    4.3 A scaled by mass at the same density (hcooh 4.3, hcooh_h2o
    4.8, cpd_ma 6.5 A).
  - `nrnet sweep` scans kpush per atom (0.015-0.03) and the bath (300,
    600 K) with two groups of 50 runs each, and writes
    `data/<system>/md_sweep.csv`: reactive share, radical share,
    candidates, literature products found (`systems/<id>/literature.yaml`)
    and group agreement. Pick the lowest kpush where the literature
    products appear and the groups agree; freeze the union of that and
    one higher setting.
  - hcho_nh3 goes first: at Zhang's settings it should reproduce their
    reactive shares, which validates the whole discovery setup.
- **D16. Each engine keeps its lowest saddle, not its first**
  (2026-11-19). A reaction's example pairs can confirm different
  saddles between the same ends: hcooh decarboxylation gave -2393
  cm^-1 at dG_act 64.9 kcal/mol from one pair and -310 cm^-1 at 93.6
  from another (1000 K), which decides whether decarboxylation or
  dehydration is faster. Every attempt now runs (the seed, when
  given, then a band on every pair); the record keeps the confirmed
  saddle with the lowest G at the system's first ranking temperature,
  and a barrierless attempt only when no pair confirms a saddle. A
  band reads as barrierless whenever its own product end lies high
  (hcooh_h2o dehydration: one pair's product end sat 56.6 kcal/mol
  above its reactant end while another pair confirmed a 74.7 saddle),
  so the checked saddle wins. `attempt_ts_energy_ev` keeps every
  confirmed attempt's energy. Seeding no longer saves
  bands; for DFT a seed-only mode is to be decided with the DFT plan.
- **D17. Literature reactions match on reacting molecules**
  (2026-11-19, discovery log F12). Whole states hide whether a water
  relays a proton or stands by, so the match compares the reacting
  molecules only.
- **D18. AIMNet2 runs the spin-aware NSE model** (2026-11-19, discovery
  log F13). `aimnet2-wb97m-d3_0` is closed-shell: it drops the
  multiplicity, so every radical was computed as a closed shell (CHO2
  -> CO2 + H at 1000 K: dG_rxn +6.7 against -33.1 with xTB and -18.8
  with UMA). `aimnet2-nse_0` reads the multiplicity, keeps AIMNet2 on
  every reaction (D14's common subset stays large) and gives HCHO within
  0.03 eV of the old model. All AIMNet2 refines rerun; the closed-shell
  results are archived.
- **D19. xTB attempts stop after 20 minutes** (2026-11-19, discovery log
  7.10). An attempt whose SCF sits near the 1000-cycle cap on every call
  (cpd_ma R014: 1.55 s per call, 3.1 h for one attempt, then
  engine_error) held back cpd_ma, and with it the ML engines that wait
  for xTB's seeds. `attempt_time_limit_s: 1200` in the xtb block of
  `configs/engines.yaml` ends such an attempt as `time_limit` (SIGALRM,
  checked between calls, so one SCF always finishes) and the search
  moves to the next attempt. Other xTB attempts take under a minute.
  The setting is per engine and off for the ML engines and DFT (ORCA
  runs as a subprocess, which the alarm would leave running).
- **D20. DFT is on hold; accuracy comes from reference data** (2026-11-19,
  user decision). Full DFT refinement would take months on this machine
  (discovery log 7.9). Engines are judged against tabulated gas-phase
  thermochemistry (reaction free energies at the ranking temperatures,
  each value with its source) and published barriers where they exist.
  Tables give reaction energies, not barriers, so barrier accuracy rests
  on the literature values alone; cpd_ma is reported as a prediction
  with an uncertainty band.
- **D21. Which ML potentials, and why; Egret removed** (2026-11-21, user
  decision). An engine here must come pretrained (there is no DFT to
  train on, D20), be trained on gas-phase molecules, have seen stretched
  bonds near saddles, and ideally take charge and spin. Not used:
  NequIP and Allegro (architectures; their released foundation models
  are for crystals, PBE), SpookyNet (no maintained general organic
  model), CHGNet and M3GNet (inorganic crystals, Materials Project,
  PBE/PBE+U), ANI-1x, 2x and 1ccx (trained near equilibrium, no charge or
  spin; AIMNet2 is their successor). Google DeepMind has no released
  molecular potential (GNoME covers crystals, DM21 is an
  exchange-correlation functional, FermiNet a wavefunction method).
  Added: ANI-1xnr as engine `ani1xnr`, trained on nanoreactor-sampled
  data (BLYP/TZV2P; H, C, N, O), so E1 asks whether reactive training
  helps. It takes no spin, so it is closed-shell only (D9) and covers 56
  of the 65 reactions; its lower DFT level is a limit, since it differs
  from the OMol25 models in data and level at once. Removed: Egret-1 and
  Egret-1t. They are little used (absent from the 2026 benchmark of ML
  potentials for transition-state searches by Marks, Vandezande and
  Gomes), closed-shell only, and slow; Egret-1t found no saddle for
  hcooh R001 in 3.6 h (discovery log 7.6). Their loaders, settings and
  tests are gone; partial results and weights are archived outside the
  repository. Discovery log 7.16.
- **D22. Duplicates are removed within a system, never across systems,
  and only from stored results** (2026-11-21, user decision; replaces
  the 2026-11-19 proposal). A reaction found by two systems stays in
  both, so no system loses what it discovered. Every unique saddle and
  species is kept; among duplicates the lowest energy is reported;
  identity is stereo-aware (enantiomers share a tag); nothing is rerun.
  One saddle leading to two products is left as found. Check of the
  stored data (discovery log 7.18): within each system nothing is left
  to remove. `freeze` already merged the 230 MD events into 65 reactions
  by their reacting molecules (hcooh 98 -> 4, hcooh_h2o 56 -> 10,
  hcho_nh3 20 -> 13, cpd_ma 56 -> 38); no two reactions in a system
  share both sides, in either direction; `refine` stores one saddle per
  reaction and one geometry per species key. The unique reactions are
  therefore the refine reaction lists, and DFT validation is chosen
  from them (next decision). Gap: 14 confirmed saddles more than
  0.1 kcal/mol from the kept one were stored as energies only (D16 kept
  the lowest geometry). They are unique by this rule but cannot be
  listed without a rerun, so they are reported as a count. For a future
  run that stores every saddle, two saddles are the same when energies
  differ by at most 0.1 kcal/mol, imaginary frequencies by at most
  10 cm^-1, and the RMSD is at most 0.1 A after the best overlay that
  may renumber atoms of one element and may mirror.

## 4. The pipeline, end to end (example: HCHO + NH3)

**Stage 1, discover (Mac, xTB binary).** `systems/hcho_nh3/` holds the
reactant geometry and `system.yaml`. We run, say, 8 replicates of
metadynamics MD at high temperature inside a spherical wall, each with
its own seed. Output: one trajectory per replicate.

**Stage 2, events.** For each frame we build the connectivity graph and
its state key. A run of identical states shorter than the minimum
lifetime (for example 50 fs) is noise and gets merged into its
neighbours. Each switch between two surviving runs is an event: the
replicate, the time, the frame before and the frame after.

**Stage 3, freeze.** Events are grouped by their unordered pair of
states, so A -> B and B -> A are one candidate reaction. For each
candidate we keep up to three example pairs (each is searched; the
lowest saddle is kept, D16). From each pair we cut out only the reacting molecules: atoms
whose bonds changed, grown to their whole molecules on both sides. Both
ends are relaxed with xTB. The candidate is dropped if the two relaxed
ends turn out to be the same state, or not the states the event named.
Survivors get SMILES and InChIKey labels and an atom-mapped reaction
SMILES (map number = trajectory atom index + 1). Everything is written
to `candidates/hcho_nh3/` with a sha256 manifest. Every later stage
checks the hash, so nobody changes the frozen set by accident.

**Stage 4, refine (once per engine).** First every species: optimize,
take the finite-difference Hessian, confirm no imaginary modes, compute
G at each ranking temperature (298 K and 600 K). Then every candidate:
optimize both ends, run a climbing-image NEB, polish the highest image
into a saddle with Sella, check for exactly one imaginary frequency, and
confirm the ends. To confirm, nudge the TS both ways along that mode,
relax, and check that we land on the two expected states. This is a
cheap stand-in for an IRC. Every engine call is logged with wall time,
device and status. Results are appended item by item, so a killed
cluster job resumes where it stopped.

**Stage 5, network.** One networkx graph per engine and system. Nodes
are states, and G(state) is the sum of its species' G. Each confirmed
reaction adds both arrows: the forward barrier, the reverse barrier
(dG_rev = dG_fwd - dG_rxn), dG_rxn, and the Eyring rate at each
temperature.

**Stage 6, rank.** R0 to R4 at both temperatures (section 6, rankers).

**Stage 7, evaluate.** The E1 to E4 tables and figures 1 to 4 are
rebuilt from `results/` with one command.

When an engine changes the story, that is data, not a crash. Say MACE
relaxes a frozen product back into the reactant, or a TS search fails.
The status is recorded and the arrow is missing from that engine's
network. E1 counts it, and E2 shows what it did to the products.

## 5. Repository layout

```text
nanoreactor-mlip-networks/
  pyproject.toml        # package nrnet; extras: xtb, mace, uma, aimnet, dft
  systems/<name>/       # reactant.xyz, system.yaml, literature.yaml
  configs/engines.yaml  # model names, fmax, NEB images, thresholds
  candidates/<name>/    # frozen species, reactions, manifest.json
  src/nrnet/
    chem.py             # connectivity, keys, labels, mapped SMILES
    nanoreactor/
      md.py             # xTB input, replicates, trajectories
      events.py         # frame states, lifetime filter, events
      candidates.py     # freeze and load the candidate set
    engines.py          # attach(atoms, settings): engine registry
    records.py          # one CSV row per engine call
    ts_search.py        # endpoints, NEB, Sella, checks
    thermo.py           # Hessian, quasi-harmonic, G(T)
    refine.py           # stage 4 runner with resume
    network.py          # states and arrows with barriers and rates
    rankers.py          # R0-R4
    kinetics.py         # rate matrix and stiff ODE solve
    evaluate.py         # E1-E4 tables
    figures.py          # figures 1-4
    cli.py              # nrnet discover|freeze|refine|network|rank|...
  tests/
  dft/                  # Slurm template with placeholders, ORCA notes
  results/  figures/  paper/writeup.tex
  Makefile              # make figures -> nrnet figures
  README.md  LICENSE
```

## 6. Modules and their public interfaces

Energies are in eV inside the code (ASE's unit). Conversion to kcal/mol
happens only in tables and figures, through one constant.

### chem

Pure functions on ASE `Atoms`; no state.

- `connectivity_graph(atoms, scale=1.2) -> nx.Graph`: bonds from ASE
  natural cutoffs times `scale`; nodes carry the element.
- `species_key(graph, atom_ids) -> str`: formula plus a short
  Weisfeiler-Lehman hash (networkx) with element labels.
- `state_key(atoms) -> tuple[str, ...]`: sorted species keys of all
  fragments.
- `reacting_atoms(graph_before, graph_after) -> list[int]`: atoms whose
  bonds changed, grown to whole molecules on both sides.
- `species_labels(atoms) -> SpeciesLabels`: SMILES and InChIKey via
  RDKit bond perception; `None` fields when perception fails.
- `mapped_reaction_smiles(before, after) -> str`: falls back to a
  connectivity-only SMILES when bond orders fail.
- `multiplicity(atoms) -> int`: from electron parity (D9).

### nanoreactor.md

- `MDSettings`: frozen dataclass read from `system.yaml` (temperature,
  time, step, dump interval, kpush, alp, wall radius, seeds, replicate
  count).
- `control_text(settings) -> str`: `$md`, `$metadyn` and `$wall` blocks.
  Kept pure so a golden-text test pins it.
- `run_replicate(reactant, settings, seed, workdir) -> Path`: runs the
  xtb binary in its own process group; on timeout or error the whole
  group is killed (`os.killpg`) inside a `finally`.
- `run_replicates(...) -> list[Path]`: process pool, as today.

### nanoreactor.events

- `Event`: frozen dataclass (replicate, time_fs, frame_before,
  frame_after, state_before, state_after).
- `stable_runs(states, min_frames) -> list[StateRun]`: run-length
  encode, then fold runs shorter than `min_frames` into their
  neighbours.
- `find_events(frames, replicate, settings) -> list[Event]`.

### nanoreactor.candidates

- `Species`, `EndpointPair`, `Candidate`: frozen dataclasses.
- `freeze(system, trajectories, out_dir) -> CandidateSet`: stage 3.
  Writes `candidates.json`, `species.extxyz`, `pairs.extxyz` and
  `manifest.json` (sha256 of each).
- `load_candidates(path) -> CandidateSet`: refuses a changed file.

### engines

One registry, `ENGINE_LOADERS`, from name to loader. The registry is
the extension point: adding an engine means one entry, nothing else.

- `attach(atoms, settings) -> Atoms` (changed from `get_calculator`
  during S2a). It returns a copy with charge and spin in `atoms.info`
  and the engine's calculator attached. Each engine takes charge and
  spin its own way (UMA and MACE read `atoms.info`, AIMNet2 and tblite
  take arguments, ORCA takes keywords). That difference stays inside
  its loader.
- ML models take seconds to load, so each model loads once per process
  (`functools.cache` on the model loader).
- Engine libraries are imported inside their loader on purpose: each
  lives in a separate environment (D5). One comment says so, the one
  allowed exception to imports-at-top.
- xTB here means tblite's ASE calculator (same GFN2 method as the MD).
  If tblite won't install on the Mac, a small ASE calculator around the
  xtb binary (`--grad`) takes its place.
- ANI-1xnr loads through torchani 2.12 `models.ANI1xnr` in float64. Its
  settings say closed-shell only, and `attach` raises
  `UnsupportedSpecies` for a doublet (D9). E1 shows whether reactive
  training helps.
- DFT means ASE's ORCA calculator at wB97M-V/def2-TZVPD (confirm the
  level on the OMol25 dataset page on Day 0).

### records

- `record_call(log_path, engine, system, task, item_id)`: a context
  manager. It times the block and writes one CSV row: wall seconds,
  status, force-call count, device, git commit and config hash. If the
  block raises, it writes the row with status `engine_error` and then
  re-raises. The row exists even when the call dies; E4 depends on it.

### ts_search

- `TSStatus` enum: `confirmed`, `neb_failed`, `saddle_not_converged`,
  `wrong_imaginary_count`, `endpoint_mismatch`, `engine_error`,
  `unsupported`.
- `TSResult`: frozen dataclass (status, ts_atoms, energy_ev,
  imaginary_cm, force_calls).
- `optimize_minimum(atoms, fmax) -> Atoms`
- `neb_guess(reactant, product, engine, search) -> Atoms`: product
  turned onto the reactant, IDPP, a plain NEB, then a climbing NEB,
  both with FIRE (see section 11 for why).
- `refine_saddle(guess, fmax) -> Atoms`: Sella, order 1.
- `count_imaginary(frequencies_cm, threshold_cm) -> int`: a mode counts
  when it is below -50 cm^-1. Smaller imaginary values are numerical
  noise.
- `confirm_endpoints(ts, mode, expected) -> bool`
- `search_candidate(candidate, engine, search, seed) -> [TSResult]`:
  with a seed (another engine's confirmed TS), tries the seeded pair
  first; then runs a band on every example pair. `refine.kept_result`
  keeps the lowest-G confirmed saddle, else a barrierless attempt (D16).

### thermo

- `vibrational_energies(atoms) -> np.ndarray`: ASE `Vibrations`,
  central differences, step 0.01 A. Drops the 6 (or 5, if linear) modes
  closest to zero, and the TS mode for a TS.
- `quasi_harmonic(frequencies_cm, floor_cm=100.0) -> np.ndarray` (D8).
- `free_energy_ev(atoms, energy_ev, vib_ev, temperature_k,
  multiplicity) -> float`: ASE `IdealGasThermo`, 1 atm, symmetry 1.

The same finite-difference code serves every engine, DFT included. One
code path means one source of numerical error, the same for all.

### refine

- `refine(system, engine) -> None`: species first, then reactions. Each
  item is appended to `results/refine/<system>/<engine>/*.jsonl`, and
  items already there are skipped on restart.

### network

- `build_network(state_g_ev, barriers, temperature_k) -> nx.DiGraph`:
  node attribute `g_ev`; edge attributes `dg_act_ev`, `dg_rxn_ev`,
  `rate_s`. Only reactions with status `confirmed` become arrows.
- `assemble.assemble_inputs(candidates, refine_paths, temperature_k)`
  reads one engine's refine results. A state's G is the sum of its
  species' G. A reaction's barrier is G_TS minus the G of its reacting
  molecules (D3), so bystanders add the same G to both ends. A reaction
  is dropped, and counted by reason, when its TS is not confirmed
  (`no_ts`) or a species on either side is not `ok`
  (`species_not_ok`). The reactant state must have a G, or the run
  stops.
- `nrnet rank --system S --engine E` builds the network at each
  temperature in `search.yaml`, runs R0 to R4, and rewrites
  `results/rank/<system>/<engine>/ranks.jsonl` (one row per ranker and
  product) and `network.jsonl` (sizes, clamps, drops). Ranking is cheap
  and deterministic, so it rewrites rather than resumes.

### rankers

Every ranker has the same signature,
`(graph, source, settings) -> list[RankedProduct]`, and lives in one
registry `RANKERS`. A `RankedProduct` holds the state, its score and the
path. A "product" is any state reachable from the reactant.

- **R0** `rank_by_free_energy`: states with dG_rxn below a threshold
  (default 0), sorted by dG_rxn. Barriers are ignored. This is the
  baseline, what the current filter does.
- **R1** `rank_by_summed_barriers`: networkx Dijkstra with weight
  dG_act. Baseline only, because adding barriers has no physical
  meaning.
- **R2** `rank_by_bottleneck`: our own Dijkstra on a heap. A path's cost
  is the highest TS on it, measured from the reactant:
  max(cost so far, G(TS) - G(reactant)). Dijkstra still works because a
  running maximum never goes down.
- **R3** `rank_by_probability`: Dijkstra with weight
  w = -ln p, where p = k_ij / sum of k_im over all arrows leaving i
  (including the one back). p is at most 1, so w >= 0, which is what
  Dijkstra needs.
- **R4** `rank_by_kinetics`: yields at `t_obs` from `kinetics`.

### kinetics

- `rate_matrix(graph, temperature_k) -> (np.ndarray, list[state])`
- `simulate(graph, source, times_s) -> np.ndarray`: `solve_ivp` with
  LSODA, the rate matrix as the Jacobian (passed as a function: an
  array crashes LSODA in scipy), and a very small `atol` (1e-20).
  Populations span 20 orders of magnitude, and a default `atol` would
  round the small ones to zero. It raises unless `status == 0`. Tested
  on the toy network: BDF stopped short of 1e12 s ("required step size
  is less than spacing") where LSODA finished.
- `boltzmann(graph, temperature_k) -> np.ndarray`: for tests.

### evaluate and figures

- E1 `step_accuracy`: MAE of dE_rxn, dG_rxn, dE_act and dG_act against
  DFT (kcal/mol), TS success rate, and TS RMSD against DFT. Atom order
  is identical by construction, so RMSD needs no reordering.
- E2 `network_agreement`: Jaccard of product sets (yield >= 1% at
  `t_obs`), Kendall tau of yields over the union, same major product,
  same top path (R3). The major product must itself reach 1%; below
  that the outcome is `no_reaction`. Without this rule a 0.0009% trace
  counted as AIMNet2's "major product" and matched xTB's 99%.
- E3 `ranker_agreement`: top-1 and top-3 agreement of R0 to R3 with R4,
  on the DFT network.
- E4 `cost`: wall time and CPU hours per network, from the call log.
- E5 `error_propagation`: how step errors become product errors. Both
  parts reuse the DFT network and the rankers; no new engine calls.
  1. Swap: give the DFT network one engine's numbers, one kind at a
     time: its barriers only, its reaction free energies only, then
     both, on DFT's arrows. R4 and the E2 metrics for each case show
     whether an engine's product errors come from barriers,
     thermodynamics or both.
  2. Noise budget: add Gaussian noise of 0.5, 1, 2 and 3 kcal/mol to
     every DFT barrier (200 seeded draws each), rerun R4, and count how
     often the major product and the top-3 set change. Each engine's
     E1 MAE placed on that curve says how accurate a potential must be
     for this network.
  Checks: zero noise, and swapping in DFT's own numbers, both return
  DFT's ranking exactly.
- `figures.py` draws figures 1 to 4 and writes tables 1 to 3 as CSV and
  LaTeX.

## 7. The toy-network test (worked by hand)

This test runs before any real data. Free energies are in kcal/mol, at
T = 298.15 K (RT = 0.5925, kT/h = 6.212e12 s^-1).

```text
states:  R = 0   A = -5   B = 0   P1 = -10   P2 = -12
arrows (TS height, absolute):
  R  <-> A    TS 15
  A  <-> P1   TS 13
  R  <-> B    TS 20
  B  <-> P2   TS  5
```

The two routes compare like this:

```text
route         summed barriers   highest TS
R -> A -> P1  15 + 18 = 33      15
R -> B -> P2  20 +  5 = 25      20
```

Expected answers:

- R0 picks **P2** (-12 beats -10).
- R1 picks **P2** (25 < 33).
- R2 picks **P1** (15 < 20).
- R3 picks **P1**: p(R -> A) is about 0.9998, p(A -> P1) about 0.967,
  while p(R -> B) is about 2e-4.
- R4 at t = 60 s picks **P1** at 0.9996 (k(R -> A) is about 63 s^-1
  and k(A -> P1) about 0.40 s^-1).
- R4 at t = 1e12 s picks **P2**, and its populations match `boltzmann`
  (P1 0.0331, P2 0.9669) within 1e-5. The P1 -> P2 leak takes about
  1e9 s.

This one network shows every claim the study makes: summing barriers
misleads, the bottleneck and probability paths agree with kinetics, and
kinetics differs from thermodynamics at a finite time. The test also
checks the solver against `scipy.linalg.expm` at 60 s. It cannot do so
at 1e12 s, where `expm` overflows to NaN. All these numbers were checked
by script on 2026-11-17.

## 8. Where things run

- **Mac (M5, 10 cores):** xTB MD (Homebrew xtb 6.7.1 is installed),
  freezing, and refinement with xTB, MACE, UMA, AIMNet2 and ANI-1xnr on
  CPU, one uv environment per engine.
- **Cluster:** DFT refinement only, with ORCA from its module and a
  Python 3.11 environment with ASE and Sella. The command is the same as
  on the Mac: `nrnet refine --system hcho_nh3 --engine dft`. The Slurm
  template in `dft/` uses placeholders: no real cluster, account or
  module names go into the public repository.
- Day 1 timing: before any TS search exists, time one wB97M-V gradient
  on a 10-atom molecule with ORCA directly. One TS costs roughly 300
  gradients (NEB, Sella, then a Hessian of 6N gradients). That estimate
  sizes the DFT subset.

## 9. Work slices

Each slice is about 15 minutes of work in the AtomField code style
(D10). The subtle parts (rankers, kinetics, TS checks) get the closest
review. Package installs happen outside the slices.

Each line: day, slice, content, and the check that closes it.

- Day 0, S0: scaffold (pyproject, ruff at 80 columns, pytest, README).
  Check: an empty `pytest` passes.
- Day 0, S1: `chem`. Check: HCHO and NH3 keys, reacting atoms, mapped
  SMILES.
- Day 0, S2: `engines` with xTB, plus one environment per ML engine.
  Check: energy and forces on HCHO from each engine.
- Day 1, S3: `nanoreactor.md`. Check: golden control text; a 1 ps smoke
  run.
- Day 1, S4: `nanoreactor.events`. Check: synthetic state sequences
  with flicker.
- Day 1, S5: `nanoreactor.candidates`, `nrnet discover` and `freeze`.
  Check: system 1 freezes; the hash check refuses an edited file.
- Day 2, S6: `records`. Check: a row is written on success and on raise.
- Day 2, S7: `ts_search`. Check: HCN -> HNC with xTB gives one
  imaginary mode and confirmed ends.
- Day 2, S8: `thermo`. Check: water's modes, the floor applied, G
  falling with T; size of the six near-zero modes per engine (Hessian
  noise).
- Day 2, S9: `refine` with resume. Check: kill and rerun skips finished
  items.
- Day 3, S10: UMA, AIMNet2, Egret-1 and ORCA loaders. Check: the
  HCN -> HNC test per engine; a doublet gives Egret `unsupported`.
- Day 4, S11: `network`, `kinetics`, `rankers`. Check: the toy test in
  section 7.
- Day 5, S12: `evaluate`, `figures`. Check: tables from a small fake
  results set.
- Day 6, S13: README, Makefile, public check. Check: a clean clone runs
  `make figures`.

Status on 2026-11-17 (evening): S0-S9, S11 and S12 are done, plus
`nrnet rank`, `--seed-from`, the ORCA loader (untested without ORCA),
README and Makefile. System 1 is refined with xTB (own band) and
AIMNet2, MACE and Egret-1 (seeded from xTB, band fallback). Open: UMA
(Hugging Face access), the DFT runs, systems 2 and 3, six functions of
16-19 lines, and a final rerun of every engine from one commit.

Added 2026-11-18, after review:

- S14: D11 barrierless detection, the clamp arrow tagged
  `barrierless`, R4 with and without. Check: HCO -> CO + H gets status
  `barrierless` and an arrow at max(G_A, G_B); a toy network with one
  barrierless arrow.
- S15: D14 common-subset rows in E1 and E2. Check: a fake results set
  where one engine lacks a radical step.
- S16: E5 swap and noise budget. Check: zero noise and a DFT-for-DFT
  swap reproduce DFT's ranking.
- S17: D13 ranking temperatures from `system.yaml`. Check: rank and
  refine read each system's own list.
- S18: D12 stereo-aware keys, before system 3. Check: endo and exo
  adducts get different keys; system 1's keys are unchanged.
- Runs: UMA (uma-s-1p2p1) and Egret-1t on system 1; DFT on the
  cluster; discover and freeze for systems 2 and 3.

The long runs start as soon as their slice lands: discovery for all three
systems after S5 (Day 1), xTB and MACE on system 1 after S9 (Day 2), DFT
for system 1 the same day.

## 10. Open items for you (Day 0)

1. **Publishing.** The new repo is public, and its nanoreactor code is a
   rewrite of afscripts parts (MD driver, event filter, NEB/Sella).
   afscripts itself stays private. Is it OK to publish this rewrite?
2. **Systems 2 and 3.** For system 2 I suggest formamide decomposition
   (to NH3 + CO, HCN + H2O, HNCO + H2), which is well studied; pick a
   paper that reports those channels. System 3 (8 to 12 heavy atoms,
   C/H/N/O) is your call.
3. **Model licenses**, checked on each model card before any run:
   - MACE foundation models are mostly under the Academic Software
     License (non-commercial). I could not confirm MACE-OMol's own
     license.
   - UMA needs its terms accepted on Hugging Face, and you need a
     Hugging Face account.
   - AIMNet2's NSE card says MIT; the main weights only say "see
     LICENSE".
   - Egret-1 is MIT (Rowan's announcement and repository).
4. **AIMNet2 and Egret-1 are not OMol25 models.** AIMNet2's default is
   trained on wB97M-D3, and Egret-1 on a wB97M-D3BJ level (confirm on
   its card), so hypothesis 1 ("OMol25-trained potentials match DFT")
   covers MACE and UMA only. AIMNet2 and Egret-1 form the "different
   training data" group. Both have reaction-trained variants. Egret-1t
   (trained on transition states) was added on 2026-11-18 as engine
   `egret_t` and runs beside Egret-1. aimnet2-rxn (registry key
   `aimnet2-rxn_0`, neutral only, D3 added after the fact) is not added
   yet.
   Models were pinned on 2026-11-18: UMA moved to uma-s-1p2p1, which the
   model card recommends over 1p2. Licenses checked the same day: MACE-OMol
   is under the Academic Software License (release notes of
   `mace_omol_0`), UMA under the FAIR Chemistry License v1, the aimnet
   package under MIT, Egret under MIT. The README lists them.
5. **Repository license**: MIT unless you prefer another.

## 11. Risks beyond the PDF's list

The report-ready record of MD runs, checks and fixes, with numbers, is
`docs/discovery-log.md`.

- **The wall radius was in the wrong unit** (found 2026-11-18). xtb
  reads `sphere:` in bohr; the code wrote the Angstrom value as is, so
  the "4 A" wall was 2.12 A (xtb prints the radius it uses). Every run
  before the fix was squeezed into a ball smaller than one HCHO + NH3
  pair, which explains why 85-95% of events stripped hydrogens and the
  concerted literature channels never appeared. Fixed in `wall_block`;
  all data from before the fix was archived, not used (D15).
- **H2 was never seen as a molecule** (found 2026-11-18). ASE's covalent
  radius for H (0.31 A) times 1.2 gives an H-H cutoff of 0.744 A, but
  GFN2-xTB's H2 bond is 0.777 A. Every H2 formed in MD read as two H
  atoms: those events were rejected as radical pairs, and CO2 + H2
  (decarboxylation) could never be recognized. `connectivity_graph`
  now uses 0.37 A for H (half of H2's bond); other bonds in our
  molecules are unchanged and system 1's species keys are the same.
  Trajectories were unaffected, so the sweeps were re-analyzed, not
  rerun.
- **One reaction per pair of states.** A candidate is keyed by its two
  states, so two mechanisms joining the same states (for example
  decarboxylation with water as a spectator and with water as a relay)
  are example pairs of one candidate, and refine keeps the first
  confirmed transition state, not the lowest. Keeping the lowest of
  several mechanisms is future work.
- **Hand-built reactions only where the MD missed them** (decided
  2026-11-18). A literature reaction whose two states the MD already
  joined only gets `literature_name` on that MD candidate; no
  hand-built geometry is added. One the MD missed is added as a new
  candidate with `source=literature`. After the fixes below, the MD
  found both decarboxylations itself; added by hand: aminomethanol ->
  CH2=NH + H2O (hcho_nh3) and the endo and exo Diels-Alder adducts
  (cpd_ma; the MD made only a [2+2] adduct).
- **xtb keeps integrating after a failed MD step** (found 2026-11-18).
  xtb 6.7.1 sometimes fails an energy evaluation inside the MD (the
  same geometry converges as a single point) and carries on, writing
  energy 0 and zero forces from then on. The atoms fly apart in
  straight lines, which the event finder read as the box atomizing:
  the commonest rejected event in hcho_nh3. `read_trajectory` now cuts
  each trajectory at its first zero-energy frame, and the sweep table
  counts such runs (`xtb_failed_replicates`). The share rises steeply
  with the push: at 300 K, 0.03 and 0.035 Eh per atom, 15% and 62% of
  hcho_nh3 runs (94% and 71% of sampling kept), 9% and 19% for hcooh,
  5% and 34% for hcooh_h2o; cpd_ma 0% at 0.01, 52% at 0.02, 78-96% at
  0.025-0.04. Trajectories were kept and re-analyzed, not rerun.

- **The wall and metadynamics settings decide what gets found.** Too
  gentle, and nothing reacts in 8 replicates. Too hard, and everything
  shatters into atoms. Day 1 tries two settings on system 1 and keeps
  the one that yields known products (the PDF's HCHO + NH3 reference).
- **Bimolecular barriers depend on the reference (D3).** Separated
  molecules give one consistent rule for every engine. The pre-reaction
  complex is recorded too, in case a reader asks.
- **Finite-difference Hessians on ML potentials can be noisy** (small
  imaginary modes). The -50 cm^-1 threshold and the quasi-harmonic
  floor handle the common cases. Modes between -50 and 0 are reported,
  not hidden.
- **UMA and AIMNet2 compute in float32** (found in S2b; MACE-OMol and
  Egret-1 run in float64). On a 3100 eV total energy, float32 rounds at
  about 3e-4 eV (0.007 kcal/mol), harmless for barriers. S8 measured
  the Hessian noise on relaxed water: the six rigid modes reach about
  50 cm^-1 for every engine, xTB included (xTB 50, MACE 50, Egret 55,
  AIMNet2 54). So the noise comes from unprojected rotations, not from
  float32, and dropping the six smallest modes by count handles it.
- **Absolute energies differ by engine** (HCHO: MACE-OMol -3115.93 eV,
  Egret-1 -3117.81, AIMNet2 -3117.83). They come from different
  reference levels, so only differences inside one engine are compared.
- **MD is reproducible in distribution only** (found 2026-11-17). The
  seed fixes the start exactly, but xtb adds about 1e-7 A of run-to-run
  noise per frame, and the chaos of hot MD grows it to whole angstroms
  within 1 ps. Rerunning `nrnet discover` gives a similar candidate set,
  not the same one. That is why candidates are frozen and committed;
  the README must say so.
- **Most hot-MD events are H stripping** (HCHO + NH3, 32 replicates:
  160 of 187 events left two radicals on one side). Closed-shell
  engines cannot describe such a diradical pair, so freezing rejects
  them as `multiple_radicals` and reports the count.
- **Kinetics describe one copy of the system.** The rate equations
  track how likely each state of the box is, not concentrations, so a
  bimolecular step has a fixed 1 atm rate and molecules from different
  copies never meet. Fine for comparing engines; stated in the limits.
- **Same step, different bystanders.** With three or more molecules,
  one elementary step appears under several whole-box states. `refine`
  searches it once (keyed by the reacting molecules alone) and reuses
  the result, marked `reused_from`.
- **Radicals break bond perception.** Species keys don't depend on it
  (D2); only labels go missing, and the run carries on.
- **The first real TS run needed a better band** (xTB, HCHO + NH3,
  2026-11-17). A 9-image climbing NEB, straight from IDPP, confirmed
  only 1 of 4 reactions. On HCHO + NH3 the band spent its images on
  the approach and the bond change fell between two images, so the
  climbing image climbed a 0.1 eV bump and Sella found a -109 cm^-1
  rotation saddle. Three changes fixed it: turn the product onto the
  reactant before interpolating, use 13 images, and run a plain NEB
  (fmax 0.5) before the climbing one (fmax 0.05). Result: aminomethanol
  formation 0.81 eV (-1482 cm^-1) and imidic acid to formamide 1.19 eV
  (-1759 cm^-1), both confirmed. The endpoint check rejected every
  wrong saddle in both runs; only the guesses were bad.
- **Some events are two steps.** Aminocarbene + H2O -> HCHO + NH3 goes
  through aminomethanol, which lived under 50 fs in the MD and so was
  filtered out. The TS found joins the carbene to aminomethanol, and
  the check reports `endpoint_mismatch`. That is correct: the event is
  not elementary. Splitting such events at the intermediate is future
  work.
- **A band can fail where the surface is fine** (Egret-1, R002,
  2026-11-17). Egret's own band rose to 4.1 eV through a region where
  an H atom comes off, ran out of climbing steps after 26 minutes, and
  its saddle fell apart into CH4NO + H. Started from xTB's TS instead,
  Egret-1 found a confirmed saddle in 23 s: 1.61 eV, -2053 cm^-1
  (Egret-1e 1.60 eV; AIMNet2's own band 1.57 eV). So `refine` takes
  `--seed-from <engine>`: the seeded pair is tried first (Sella from
  the seed, then the same frequency and endpoint checks), and the
  engine's own band runs only if that fails. Every engine still finds
  and confirms its own saddle on its own surface; the seed picks the
  start. Each reaction row records `route` (band or seed) and
  `seed_engine`, so E1 can report band and seeded success apart. DFT
  needs this anyway: a 13-image band costs thousands of gradients.
- **Barrierless steps have no saddle.** HCO -> CO + H (a bond breaking
  into two radicals) has its band maximum at the product end. The
  first run "confirmed" a flat -71 cm^-1 saddle there, which was noise;
  the second correctly finds none (`wrong_imaginary_count`). Decided
  on 2026-11-18: such steps get the clamp rule, D11 (slice S14).
- **The MD misses reactions that need a set approach or come second**
  (2026-11-18). From HCHO + NH3 starts it makes aminomethanol but never
  dehydrates it (a second step, barrier about 55 kcal/mol, in the same
  10 ps run). From the stacked cyclopentadiene + maleic anhydride
  complexes it never makes either Diels-Alder adduct, at any push from
  0.01 to 0.04 Eh per atom: the RMSD push moves the box away from where
  it has been, in any direction, not toward a face-to-face approach.
  These enter as flagged literature reactions and are reported apart
  from MD discoveries. Later work, agreed with the user: extra runs
  started from the intermediates (aminomethanol; the endo and exo
  complexes held closer, or a tighter cpd_ma wall), then keep a
  flagged reaction only if the MD still misses it.
- **Endpoints taken at the crossing were often thrown away** (found
  2026-11-18 by an independent product census: one frame per ps of
  every run relaxed with GFN2-xTB and its molecules listed). The census
  found a Diels-Alder adduct in cpd_ma (0.01 Eh per atom, rep_18) that
  no frozen set held. The event finder had seen it; the check then
  rejected it, because the "before" frame was the last one before the
  change, already on the barrier, and relaxed into the adduct
  (`state_mismatch`). That rejected 18 of 21 cpd_ma events at 0.01.
  `find_events` now takes each endpoint `min_frames` (50 fs) into its
  stable run, in that run's own state. Otherwise the census agreed with
  the frozen sets: what they lacked were radicals and species seen in
  one run only. Script: the census should become an `nrnet` check.
