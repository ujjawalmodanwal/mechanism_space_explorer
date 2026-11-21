# Discovery log: MD settings, checks and fixes

This file records how the reaction candidates were found: every MD
run, every check run on the results, every fault the checks exposed,
and what each fix changed. Section 7 does the same for refinement.
It is written for the report; PLAN.md section 11 holds the same
findings as risks and decisions.

All MD and metadynamics runs finished on 2026-11-18. Everything after
that is analysis of the saved trajectories, which are never edited.

## 1. MD runs

Engine: xtb 6.7.1, GFN2-xTB. Each replicate starts from a frame of an
unbiased equilibration (100 ps at the bath temperature, inside the
wall, after an xtb optimization of the start structure), then runs
10 ps of RMSD metadynamics.

| Setting | Value | Why |
|---|---|---|
| Bath | 300 K (600 K in the sweep only) | Zhang, Xu, Lan, JCTC 2023 (HCHO + NH3) |
| Step, dump | 0.5 fs, 5 fs | no SHAKE in biased runs; events need < 10 fs |
| Bias | RMSD metadynamics, alp 0.7 bohr^-2, save 1 | Zhang 2023; Grimme 2019 |
| Push (kpush) | per atom x atoms, scanned | Grimme's 0.02 x N rule, then the sweep |
| Wall | logfermi sphere, xtb defaults (6000 K, beta 10 bohr^-1) | Zhang 2023 |
| Wall radius | hcho_nh3 4.3 A, hcooh 4.3 A, hcooh_h2o 4.8 A, cpd_ma 6.5 A | Zhang's 4.3 A, scaled by mass at equal density |
| Replicates | 50 per group | |

Runs, all at 10 ps:

| System | Sweep groups | Production | Biased runs |
|---|---|---|---|
| hcho_nh3 | 300 K and 600 K x 0.015-0.035 per atom, 1 group | 300 K, 0.03 and 0.035, 6 groups | 1000 |
| hcooh (trans and cis starts) | same | same | 1000 |
| hcooh_h2o | same | same | 1000 |
| cpd_ma (endo and exo starts) | 300 K x 0.01-0.04, 1 group | from the sweep: 0.01, 0.015, 0.02 | 350 |

Production groups 1-4 are the main runs and 5-6 the check runs
(section 3.1). Wall times on 10 cores, one xtb thread per run:
production 13-34 min per system; cpd_ma 1.9 h over its two sweeps.

## 2. Faults found and fixed

In the order found. "Rerun" means the MD was run again; "re-analyzed"
means only event detection and freezing were redone.

| # | Fault | Found by | Fix | Effect |
|---|---|---|---|---|
| F1 | xtb reads the wall radius in bohr; we wrote Angstrom, so "4.0" was a 2.12 A wall | reading xtb's input rules | `wall_block` converts to bohr | all earlier data archived; all MD rerun |
| F2 | `xtb --opt` crashes in some builds (Fortran format error, optimizer.f90:852) | failed start optimizations | `$opt engine=lbfgs` | start structures optimize |
| F3 | every group started from the same equilibration frames | identical group results | seeded random frame draw, seed per group | group 1 rerun at production settings |
| F4 | group overlap compared atom-numbered SMILES, which differ between runs | overlap 0 for identical reactions | compare unordered state pairs | overlap values valid |
| F5 | H-H cutoff 0.744 A (ASE 0.31 A x 1.2) below xTB's H2 bond, 0.777 A: H2 read as two H atoms | decarboxylation never found | H radius 0.37 A, scaled like the others | re-analyzed; hcooh now finds HCOOH -> CO2 + H2 68 times |
| F6 | xtb keeps integrating after a failed MD step, writing energy 0 and zero forces; atoms fly apart, read as atomization | bond-cutoff audit (3.2) | trajectories cut at the first zero-energy frame; sweep CSV counts `xtb_failed_replicates` | re-analyzed; see table below |
| F7 | event endpoints taken at the crossing; the "before" frame sits on the barrier and relaxes to the product, so the check rejects the event | product census (3.3) | endpoints taken 50 fs into each stable run, in its own state | re-analyzed; cpd_ma at 0.01 had 18 of 21 events rejected; after the fix its reactions kept went 3 -> 7 and the MD's exo Diels-Alder adduct appears |
| F8 | a matched literature pair replaced an MD pair | reviewing the literature code | MD-found literature reactions get only a label | see section 4 |

F6, runs broken by xtb and sampling kept after the cut (300 K):

| System | 0.025 | 0.03 | 0.035 |
|---|---|---|---|
| hcho_nh3 | 3/50 | 44/300 (94% kept) | 186/300 (71% kept) |
| hcooh | 0/50 | 28/300 (96%) | 56/300 (87%) |
| hcooh_h2o | 0/50 | 15/300 (98%) | 102/300 (86%) |
| cpd_ma | 39/50 (49%) | 46/50 (33%) | 47/50 (21%) |

cpd_ma at 0.01, 0.015 and 0.02: 0, 2 and 26 of 50 broken. A single
point on the frame where xtb broke converges in 14 SCF cycles, so the
fault is inside xtb's MD step, not the geometry; atoms had been thrown
about 1.2 A past the wall just before. Most runs break after 5 ps.

Consequence: the first sweep's agreement with Zhang's 12% reactive
runs (HCHO + NH3, 0.20 Eh) came from broken runs. After F5-F7 that
push gives no reactive run in 50. We cannot tell from the paper
whether its count screened out such runs.

## 3. Checks

### 3.1 Main versus check groups

Do 100 independent runs (other start frames and seeds) find what the
200 main runs found? Production, both pushes together:

| System | Main reactions seen 2+ times | Also in check | Single-event reactions not shared |
|---|---|---|---|
| hcho_nh3 | 2 | 2 | 8 |
| hcooh | 3 | 2 | 2 |
| hcooh_h2o | 5 | 4 | 4 |

The main channels agree in both (hcooh decarboxylation 44 main / 24
check, dehydration 17 / 8); the two misses were seen 3-4 times in the
main runs, near chance (a reaction seen 4 times in 200 runs has about
a 1 in 8 chance of not appearing in 100). Rare reactions are not
converged: each 100 runs add 1-3. The production sets use all 600
runs. Numbers after F5-F7; results in
`data/<system>/checks/group_comparison.csv`. Script:
`scripts/checks/compare_groups.py`.

### 3.2 Bond-cutoff audit

Events in all 600 hcho_nh3 production runs at three cutoffs (bond if
distance < scale x sum of covalent radii), graph only:

| Scale | Events | With 2+ radicals |
|---|---|---|
| 1.1 | 595 | 58% |
| 1.2 (used) | 488 | 80% |
| 1.3 | 458 | 72% |

1.1 adds flicker (H leaving formaldehyde and returning, 65 + 43
times); 1.3 changes little. The kept reactions are the same at all
three, so the cutoff is not why reactions are missing. The commonest
rejected event, the whole box into atoms (138), exposed F6. Run before
F6 and F7. Script: `scripts/checks/bond_audit.py`.

### 3.3 Product census

Independent of the event finder: one frame per ps of every run
(stopping at xtb breaks) is optimized with GFN2-xTB and its molecules
listed, then compared with every frozen set. About 18,000
optimizations, 3 minutes. Before F7:

- hcho_nh3, hcooh, hcooh_h2o: the frozen sets held every molecule
  except radicals (rejected by design) and molecules seen in one run
  (HCN, HNC, HNCO, a nitrone). Methanimine from aminomethanol never
  appears.
- cpd_ma: a Diels-Alder norbornene adduct in one run (0.01, rep_18,
  from 130 fs to the end) that no frozen set held. This exposed F7.

After F5-F7 (final sets): hcho_nh3, hcooh and hcooh_h2o hold every
molecule seen in two or more runs except two carbenes, aminohydroxy-
carbene and dihydroxycarbene (2 runs each). cpd_ma (census over all
350 runs, sets from 0.01-0.02 only): both Diels-Alder adducts are in
the final set; the molecules it lacks are radicals (C4H3O3, C5H5,
C2H3, C3HO; rejected by design) and products of the 0.025-0.04 runs,
which the production set does not use. Results in
`data/<system>/checks/census.csv`. Script: `scripts/checks/census.py`.

### 3.4 Endpoint mismatches in cpd_ma

After F7, cpd_ma still rejects 58 of its 111 production events because
an optimized end changes identity. Classified (30 after-ends, 28
before-ends): almost all are loose maleic anhydride-cyclopentadiene
contacts that count as bonded in the hot MD (C9H8O3 keys 4569, 9dec,
b612) and fall apart into the two molecules when optimized, and
distorted cyclopentadienes that relax to the normal one. These are
correct rejections: the MD state is not a minimum. None is a
Diels-Alder adduct.

## 4. Literature reactions

Rule (decided 2026-11-18): a literature reaction whose two states the
MD already joined only gets `literature_name` on that MD reaction. One
the MD missed is added as a hand-built reaction with
`source=literature`, and reported apart from MD discoveries.

| System | Reaction | MD found it? |
|---|---|---|
| hcooh | decarboxylation, HCOOH -> CO2 + H2 | yes, after F5 (labelled) |
| hcooh_h2o | water-assisted decarboxylation | yes, after F5 (labelled) |
| hcho_nh3 | aminomethanol -> CH2=NH + H2O | no: hand-built from the MD's own aminomethanol |
| cpd_ma | endo Diels-Alder | no: hand-built |
| cpd_ma | exo Diels-Alder | yes, after F7: 1 event at 0.01 (labelled; face checked, carbonyl carbons 2.75 A from the CH2) |

Hand-built geometries: `systems/*/lit_*.xyz`, same atom order as the
start file; Diels-Alder adducts made by an xtb optimization with the
two new C-C bonds held at 1.56 A, then a free one, and checked for
face (endo: carbonyl carbons 2.71 A from the C=C, 3.51 A from the CH2).

## 5. Known limits, later work

- The MD misses reactions that need a set approach or come second:
  aminomethanol dehydration (about 55 kcal/mol, after addition in the
  same run) and, at most pushes, the Diels-Alder. Later: runs started
  from the intermediates, then keep a flagged reaction only if the MD
  still misses it.
- Rare reactions are not converged (3.1).
- Events with two or more radicals are rejected because closed-shell
  engines cannot treat them; most of hcho_nh3's activity is of this
  kind.
- The census should become an `nrnet` check run after every freeze.

## 6. Final candidate sets

Frozen 2026-11-18 after F5-F7, in `candidates/<system>/` (format:
README). These are what every engine refines.

| System | Runs | Events | Reactions (MD + hand-built) | Molecules | Rejected: 2+ radicals / mismatch |
|---|---|---|---|---|---|
| hcho_nh3 | 600 (0.03, 0.035) | 173 | 13 (12 + 1) | 15 | 153 / 0 |
| hcooh | 600 (0.03, 0.035) | 652 | 4 (4 + 0) | 8 | 554 / 0 |
| hcooh_h2o | 600 (0.03, 0.035) | 184 | 10 (10 + 0) | 11 | 128 / 0 |
| cpd_ma | 150 (0.01, 0.015, 0.02) | 111 | 38 (37 + 1) | 44 | 4 / 58 |

Main reactions, events in brackets:

- hcho_nh3: HCHO + NH3 -> aminomethanol (7), -> CO + H2 + NH3 (3);
  hydroxycarbene + NH3 -> aminomethanol (1); HCHO + NH3 -> CH2=NH +
  H2O directly (1); hand-built aminomethanol -> CH2=NH + H2O.
- hcooh: decarboxylation (68), dehydration (25), two radical steps.
- hcooh_h2o: dehydration (21), water-assisted dehydration (3; the
  water is among the reacting molecules), orthoformic acid HC(OH)3
  (11), decarboxylation beside a bystander water (10), water-assisted
  decarboxylation (1, labelled; see F12 in 7.3), carbonic acid + H2
  (3).
- cpd_ma: the commonest product (12) is not a Diels-Alder adduct but a
  one-bond zwitterion, a maleic anhydride carbonyl O bonded to a
  cyclopentadiene C; likely an xTB artefact, which refinement with the
  other engines will test. Also a [2+2] adduct, ring-opened and
  CO2-loss products, the exo Diels-Alder adduct (1), and the
  hand-built endo one. cpd_ma's production push was chosen from the
  sweep, not run separately: 0.01-0.02, where few runs break.

## 7. Refinement

Each engine refines the four frozen sets with `nrnet refine`.
`scripts/checks/refine_summary.py` writes one row per reaction and
ranking temperature to `data/<system>/checks/refine_<engine>.csv`:
status, route, imaginary frequency, forward and reverse barriers and
dG_rxn in kcal/mol, using the network's rule (G_TS minus the G of the
reacting molecules only).

### 7.1 Faults found and fixed

Numbering continues from section 2. The first xTB pass (2026-11-19)
ran at 2 threads per system, the resume at 4; its results are in
`~/mse-archive-2026-11-18/refine_xtb_mixed_threads/`.

| # | Fault | Found by | Fix | Effect |
|---|---|---|---|---|
| F9 | Sella raises `RuntimeError` ("Geometry update ODE is taking too long to converge!") when its internal-coordinate step will not solve; the error stopped the whole refine | hcooh_h2o and cpd_ma refines crashed (cpd_ma after 2.8 h, 24 of 38 reactions) | `saddle_step` records the attempt as `saddle_not_converged` with Sella's message and moves to the next pair; engine failures (`CalculationFailed`) still pass through | both runs finished; one cpd_ma reaction records this status |
| F10 | tblite's default of 250 SCF cycles is too few on stretched geometries | 16 of 65 reactions ended `engine_error` (hcho_nh3 6, hcooh_h2o 2, cpd_ma 8) | `max_iterations: 1000` for xtb in `configs/engines.yaml`; converged results are unchanged | retry of the 16: 9 now end on a real search status (5 endpoint_mismatch, 4 wrong_imaginary_count), none confirmed; 7 still fail at 1000 cycles. Stronger mixer damping (0.1) was tried on 4 and rejected: it lands on different SCF solutions (R007 confirmed -> endpoint_mismatch) |
| F11 | the thread count changes outcomes on borderline geometries | R002 and R007 gave different statuses in a 1-thread probe and the 2-thread run | xTB refines run with `OMP_NUM_THREADS=1`; all four sets rerun from scratch | see the table below; 1 thread is also 5x faster on these molecules |

F11, two hcho_nh3 reactions repeated at each thread count (1000 SCF
cycles). Each status repeated exactly at a fixed count:

| Threads | R002 | R007 | R007 wall time |
|---|---|---|---|
| 1 (twice) | endpoint_mismatch | confirmed | 2.3 s |
| 2 (twice) | engine_error (SCF) | endpoint_mismatch | 10.6-11.9 s |
| 4 | endpoint_mismatch | confirmed | 11.1 s |

### 7.2 xTB results (1 thread, 1000 SCF cycles)

Rerun from scratch on 2026-11-19, all four systems in parallel, one
thread each. Molecules: every one is a true minimum except one cpd_ma
species that relaxes into another (`endpoint_changed`).

| System | Reactions | Confirmed | Barrierless | Endpoint mismatch | Wrong imaginary count | Endpoint changed | SCF failure | Wall time |
|---|---|---|---|---|---|---|---|---|
| hcho_nh3 | 13 | 4 | 1 | 5 | 1 | 1 | 1 | 12 s |
| hcooh | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 105 s |
| hcooh_h2o | 10 | 6 | 0 | 2 | 0 | 0 | 2 | 116 s |
| cpd_ma | 38 | 7 | 4 | 8 | 11 | 3 | 5 | 3.4 h |

SCF failures left at 1000 cycles: hcho_nh3 R008; hcooh_h2o R004
(water-assisted dehydration, 3 events) and R006 (HCOOH + H2O ->
carbonic acid + H2); cpd_ma R007, R014, R028, R029, R034.

Free-energy barriers against the literature (kcal/mol):

| System | Reaction | xTB dG_act | Literature |
|---|---|---|---|
| hcho_nh3, 298 K | HCHO + NH3 -> aminomethanol (R004) | 26.4 | 34; 31.4 (Zhang, B3LYP-D3/def2-SVP) |
| hcho_nh3, 298 K | aminomethanol dehydration (R013) | 60.8 | 55.0-55.9 (230-234 kJ/mol) |
| hcooh, 1000 K | decarboxylation (R001) | 93.6, see 7.3 | similar to dehydration |
| hcooh, 1000 K | dehydration (R002) | 74.7 | dominant in gas phase |
| cpd_ma, 298 K | Diels-Alder endo (R038) / exo (R008) | 8.0 / 10.3, endo by 2.3 | endo by 2.2 |

The literature values may be energies or enthalpies, not free
energies at these temperatures; sources are still to add.

### 7.3 First confirmed saddle, and an assisted-reaction label

Both found in the 1-thread results of 7.2 and decided on 2026-11-19
(PLAN D16, D17). The 7.2 results are archived in
`~/mse-archive-2026-11-18/refine_xtb_first_confirmed/`.

- **The search kept the first confirmed pair, not the lowest saddle.**
  hcooh R001 (decarboxylation) has two confirmed saddles joining the
  same ends. Pair 1: -2393 cm^-1, dE_act 65.0, dG_act 64.9 (the
  four-centre H2 elimination; first xTB pass, where pair 0 hit the
  250-cycle cap). Pair 0: -310 cm^-1, dE_act 115.6, dG_act 93.6 (the
  1-thread rerun, where pair 0 converges). Which saddle is kept decides
  whether decarboxylation or dehydration (74.7) is the faster channel.
  Fix (D16): every attempt runs and the record keeps the lowest-G
  confirmed saddle; `attempt_ts_energy_ev` keeps each confirmed
  attempt's energy. Under the new rule all three hcooh R001 pairs
  confirm: pairs 1 and 2 the same -2393 cm^-1 saddle, pair 0 one
  43.6 kcal/mol higher in energy; pair 1 is kept (dG_act 64.9).
- **A barrierless band first beat a confirmed saddle (fixed the same
  day).** The first version of D16 let a barrierless attempt win. On
  hcooh_h2o R003 (dehydration), pair 0 confirmed a saddle at dG_act
  74.7, but pair 2's product end had relaxed to a structure 56.6
  kcal/mol above its reactant end (separated CO + H2O lie at dG_rxn
  +19.5), its band rose steadily to that end, and the step read as
  barrierless; the clamp rule would then have set the barrier to 19.5.
  Fix: a confirmed saddle beats a barrierless band, and a barrierless
  attempt no longer ends the search. The run with the faulty rule was
  stopped and all four sets rerun.
- **F12: an assisted reaction was matched by whole states.**
  `candidate_matches_states` compared full states, so hcooh_h2o's MD
  decarboxylation with water as a bystander (R001, reacting molecules
  HCOOH -> CO2 + H2, 10 events, dG_act 93.57, the same as bare HCOOH)
  took the label `decarboxylation_water_assisted`. Fix (D17): match on
  the reacting molecules. Re-freezing hcooh_h2o moved the label to
  R002, an MD event (1) whose reacting molecules include the water
  (HCOOH + H2O -> CO2 + H2 + H2O), so under the label-only rule
  nothing is added. Reactions, species and keys are otherwise
  unchanged; three pair geometries moved by under 0.001 A and one
  (R005, a doublet) by 0.17 A, the xTB relaxation not repeating
  exactly. The old set is in
  `~/mse-archive-2026-11-18/candidates_hcooh_h2o_before_F12/`.

### 7.4 First ML results against xTB (partial, 2026-11-19)

Done by 09:20: AIMNet2 on the three small systems, UMA on hcooh. MACE,
Egret-1 and Egret-1t are still on hcooh, UMA on hcooh_h2o. Six jobs share
the CPUs, so wall times are not comparable yet. Each ML engine runs
`--seed-from xtb`: the xTB saddle is tried first, then the engine's own
band on every pair, and D16 keeps the lowest confirmed saddle.

**F13: AIMNet2 ignores spin (fixed: PLAN D18, AIMNet2-NSE).**
`aimnet2-wb97m-d3_0` is a closed-shell model (`is_nse` is False). aimnet
0.2.6 drops `mult` with a warning ("Input mult=[2.0] is ignored ... use
an NSE model"), and all three logs show it. Every doublet (CHO2, H, CHO,
CH4NO, ...) is therefore computed as the closed-shell state. Example,
CHO2 -> CO2 + H at 1000 K: dG_rxn +6.7 and +20.5 (hcooh R003, R004)
against -33.1 and -25.3 with xTB and -18.8 and -32.3 with UMA.
MACE-OMol25 reads `atoms.info["spin"]` as `total_spin`, and UMA has
`spin` in its `r_data_keys`; Egret is already `closed_shell_only`.
Fix: AIMNet2 now runs `aimnet2-nse_0` (same registry, sha256
b98b3449...). Check: O2 singlet and triplet give the same energy under
the old model (-4091.529 eV both) and differ under NSE (triplet 1.37 eV
lower); HCHO moves by 0.03 eV. `test_aimnet_engine_reads_spin` guards
it. All AIMNet2 refines rerun; the closed-shell results and logs are in
`~/mse-archive-2026-11-18/refine_aimnet_closed_shell/`. The AIMNet2
columns below are from the old model, kept as the record of the fault.

**Seeds.** Counts per engine and system. "Seed higher by" compares the
seed's saddle with the best band saddle when both confirmed.

| Engine, system | xTB seeds | Seed confirmed | Saddle only from the seed | Seed higher by (kcal/mol) |
|---|---|---|---|---|
| AIMNet2, hcooh | 4 | 3 | 1 (R003, radical) | 0.0 (R001, R002) |
| AIMNet2, hcooh_h2o | 6 | 6 | 2 (R005, R007, radicals) | 0.0 (R001, R003, R008, R010) |
| AIMNet2, hcho_nh3 | 4 | 2 | 1 (R004) | 0.5 (R013) |
| UMA, hcooh | 4 | 4 | 0 | 0.0 (all four) |

- Whenever the seed and a band both confirmed, they reached the same
  saddle (within 0.5 kcal/mol). The seed does not bias the result.
- The seed matters when the engine's own bands fail. In hcho_nh3 R004
  (HCHO + NH3 -> aminomethanol), AIMNet2's three bands ended
  wrong_imaginary_count, wrong_imaginary_count, endpoint_mismatch; only
  the seed gave a saddle. The other seed-only cases are radicals (F13).
- A seed can fail where xTB succeeded: AIMNet2 hcho_nh3 R006 (seed
  wrong_imaginary_count; all three bands confirm) and R007 (seed
  saddle_not_converged; the band confirms).
- The kept saddles stay close to xTB's (Kabsch RMSD 0.04-0.15 A), except
  AIMNet2 hcho_nh3 R006 (0.33), R007 (0.82), R013 (0.33) and UMA hcooh
  R002 (1.10; UMA's seed and bands reach the same saddle).
- Pairs still disagree within one engine: the confirmed attempts span
  17.9 kcal/mol for UMA hcooh R003 and 6.5 for AIMNet2 hcho_nh3 R006, so
  D16 (search every pair) is needed for the ML engines too.
- AIMNet2 hcooh_h2o R008 confirms a saddle but has no free energies: its
  product HC(OH)3 (CH4O3-1b786c83) relaxes to an `imaginary_minimum`.

**Barriers, closed-shell reactions (dG_act, kcal/mol).**

| Reaction | T (K) | xTB | AIMNet2 | UMA | Literature |
|---|---|---|---|---|---|
| hcooh R001, decarboxylation | 1000 | 64.9 | 79.4 | 68.3 | similar to dehydration |
| hcooh R002, dehydration | 1000 | 74.7 | 74.0 | 63.3 | dominant in gas phase |
| hcho_nh3 R004, HCHO + NH3 -> aminomethanol | 298 | 26.4 | 45.2 | | 34; 31.4 |
| hcho_nh3 R006, HCHO -> CO + H2 | 298 | 68.1 | 101.2 | | |
| hcho_nh3 R013, aminomethanol dehydration | 298 | 60.8 | 58.1 | | 55.0-55.9 |

xTB alone makes HCOOH decarboxylation the faster channel. AIMNet2 (by
5.4) and UMA (by 5.0) make dehydration faster, as the literature says.
AIMNet2 repeats its hcooh barriers in hcooh_h2o (79.35, 73.97).

**Check: reaction free energies against gas-phase tables.** Reference
from 298 K enthalpies of formation and entropies (NIST/JANAF). Heat
capacity changes are ignored, worth a few kcal/mol at 1000 K.

| Reaction | T (K) | Tables | xTB | AIMNet2 | UMA |
|---|---|---|---|---|---|
| HCOOH -> CO2 + H2 | 1000 | -26.5 | -38.2 | -32.3 | -30.0 |
| HCOOH -> CO + H2O | 1000 | -26.7 | +19.5 | -27.3 | -27.0 |
| HCHO -> CO + H2 | 298 | -8.3 | +31.0 | -8.5 | |
| HCHO -> CO + H2 | 600 | -16.2 | +22.6 | -16.9 | |

xTB puts every CO-forming step 39-46 kcal/mol too high; AIMNet2 and UMA
land within 1 kcal/mol. Open question: whether this error suppressed CO
channels in the xTB nanoreactor MD itself.

### 7.5 AIMNet2-NSE rerun (PLAN D18, 2026-11-19)

The three small systems, rerun with `aimnet2-nse_0`, seeded from xTB, 1
thread each, all three in parallel beside four other jobs. Wall time:
hcooh 706 s, hcooh_h2o 1755 s, hcho_nh3 1500 s. No spin warnings; every
species is a true minimum (8, 11 and 15). Confirmed saddles: hcooh 4 of
4, hcooh_h2o 8 of 10, hcho_nh3 6 of 13 (closed-shell model: 3, 8, 4).

**Radicals now agree in sign with xTB and UMA** (1000 K, kcal/mol):

| Reaction | dG_act xTB / old / NSE / UMA | dG_rxn xTB / old / NSE / UMA |
|---|---|---|
| HOCO -> CO2 + H (hcooh R003) | 2.1 / 17.7 / 31.4 / 26.3 | -33.1 / +6.7 / -26.2 / -18.8 |
| HCOO -> CO2 + H (hcooh R004) | 0.3 / barrierless / 9.2 / 2.0 | -25.3 / +20.5 / -41.5 / -32.3 |

The NSE model and UMA put HCOO lower in dG_rxn than HOCO (by 15.3 and
13.6); xTB reverses the order. xTB lets HOCO fall apart almost freely
(2.1); both ML engines give it a real barrier. hcooh and hcooh_h2o now
give the same AIMNet2 numbers for the same reactions (79.55/79.59,
79.06/79.06, 31.40/31.43, 9.21/9.21); the closed-shell model had given
HCOO -> CO2 + H barrierless in one and 27.0 in the other.

**Closed-shell barriers also moved, by up to 5 kcal/mol** (dG_act):

| Reaction | T (K) | xTB | old | NSE | UMA | Literature |
|---|---|---|---|---|---|---|
| HCOOH decarboxylation (R001) | 1000 | 64.9 | 79.4 | 79.6 | 68.3 | similar to dehydration |
| HCOOH dehydration (R002) | 1000 | 74.7 | 74.0 | 79.1 | 63.3 | dominant in gas phase |
| HCOOH + H2O -> HC(OH)3 (hcooh_h2o R008) | 1000 | 67.4 | no G | 83.1 | | |
| HCHO + NH3 -> aminomethanol (R004) | 298 | 26.4 | 45.2 | 43.4 | | 34; 31.4 |
| HCHO -> CO + H2 (R006) | 298 | 68.1 | 101.2 | 103.2 | | |
| aminomethanol dehydration (R013) | 298 | 60.8 | 58.1 | 58.5 | | 55.0-55.9 |

Correction to 7.4: with the NSE model, AIMNet2's two HCOOH channels are
tied (dehydration lower by 0.5), not 5.4 apart. UMA still makes
dehydration faster (by 5.0) and xTB decarboxylation (by 9.8).

**Reaction free energies against the tables** (7.4 check, NSE added,
plus HCO -> H + CO from the same tables):

| Reaction | T (K) | Tables | xTB | NSE | UMA |
|---|---|---|---|---|---|
| HCOOH -> CO2 + H2 | 1000 | -26.5 | -38.2 | -29.5 | -30.0 |
| HCOOH -> CO + H2O | 1000 | -26.7 | +19.5 | -27.8 | -27.0 |
| HCHO -> CO + H2 | 298 | -8.3 | +31.0 | -5.9 | |
| HCO -> H + CO (hcho_nh3 R001) | 298 | +9.0 | +39.9 | +1.9 | |

The xTB CO error holds for the radical step too (+30.9 against the
tables). NSE is within 3 kcal/mol on the closed-shell rows and 7.1 low
on HCO.

**Seeds under the NSE model.** The xTB seed confirmed in 11 of 14 tries
(hcooh R001 saddle_not_converged; hcho_nh3 R006 wrong_imaginary_count,
R007 endpoint_mismatch; each still confirmed by a band). Only hcooh_h2o
R010 (H2CO3 -> CO2 + H2O) got its saddle from the seed alone. The
radical "seed-only" cases of 7.4 were artifacts: the bands now confirm
them. Seed and band now sometimes reach different saddles, and the
energy order can differ from the G order at 1000 K: in hcooh R002 and
hcooh_h2o R003 the seed's saddle is 2.4 kcal/mol higher in energy but
lower in G, so D16 keeps it; in hcooh_h2o R001 the seed's is 4.1 lower
in energy and a band's is kept. D16 ranks by G as intended.

### 7.6 Engine cost, and a force-call undercount (2026-11-19)

**F14: `force_calls` counts optimizer steps, not calculator calls (small
for AIMNet2; corrected the same day).** Each stage adds its step count
(`steps`, `steps x (images - 2)` for the band, `6 x atoms` for the
frequencies); Sella makes extra calls inside a step. Counting real
`Calculator.calculate` calls (7.7) gives 1.03x the count for band
attempts and 1.25x for seed attempts (AIMNet2-NSE). A first estimate
here said about 3x, from timed single calls (UMA 0.027 s, MACE 0.277 s
on the hcooh R001 saddle) against wall time per counted call in the runs
(UMA 0.081 s, MACE 0.88 s); that gap is real but is not the counter, and
its cause (CPU sharing between 8 jobs, or per-call cost inside the
search) is open. Cost comparisons (E4) use wall time.

**Cost per engine so far (hcooh, 4 reactions):** AIMNet2-NSE 12 min,
UMA 51 min, MACE 1.9 h for R001 alone (7949 counted calls; dG_act 67.8,
-2321 cm^-1, seed route, 0.06 A from xTB's saddle). MACE-OMol25
extra_large is about 10x slower per call than UMA, and threads do not
help on these molecules (0.28, 0.30, 0.31 s at 1, 2, 4 threads). Egret-1
and Egret-1t had not finished R001 after 2.3 h despite ~0.1 s per
molecule-optimization call, which points to searches running to their
step caps; to check when their first reaction lands.

MACE reaction free energies (species only, all four hcooh reactions):
-29.9, -27.1, -19.1, -33.1 kcal/mol, within 0.8 of UMA's (-30.0, -27.0,
-18.8, -32.3): the two OMol25 models agree closely on thermochemistry.

Egret R001 (hcooh decarboxylation), 2026-11-19: Egret-1 confirmed from
the seed (0.07 A from xTB's saddle; bands saddle_not_converged,
confirmed, confirmed) in 2.9 h; Egret-1t, the TS-trained model, found
no saddle on any attempt (seed saddle_not_converged; bands
wrong_imaginary_count, endpoint_mismatch, endpoint_mismatch) in 3.6 h.
Both spent about 0.7 s per counted call against 0.25 s per call in
their seed stages. Not the cause of the gap: switching the calculator
between 11 band images (351 ms per MACE call) costs the same as reusing
one structure (359 ms). xTB on cpd_ma: R014 (an SCF failure in 7.2) has
run 4.5 h without finishing, at full CPU, mostly in BLAS. In the
first-confirmed run R014 took 11176 s for one attempt (7193 counted
calls, 1.55 s per call, against about 3 ms on cpd_ma's other
reactions: SCF near the 1000-cycle cap on every call) and ended
engine_error; D16 now searches every pair. R028 and R034 behaved the
same way (7417 and 7059 calls before engine_error).

### 7.7 Seed versus band: cost and reliability (2026-11-19)

Each attempt of the AIMNet2-NSE refine was rerun on its own and timed,
with real `Calculator.calculate` calls counted (three systems in
parallel, 1 thread each, eight jobs on the machine). Every status
repeated the 7.5 refine exactly.

| Attempt | Count | Confirmed | Median wall time | Median real calls |
|---|---|---|---|---|
| Seed (Sella from xTB's saddle) | 14 | 11 (79%) | 5 s | 262 |
| Band (own NEB on one pair) | 49 | 27 (55%) | 84 s | 7050 |

Pooled over AIMNet2-NSE, UMA and MACE records (21 seeded reactions),
the seed confirmed 18 of 21 tries (86%) and a single band 40 of 69
(58%). Where both confirmed (17 reactions), they reached the same saddle
in 13; in 4 they differ by 2.4-4.2 kcal/mol in energy, and D16 kept the
band's saddle in 2 (hcooh_h2o R001, hcho_nh3 R013). The seed failed on 3
reactions that a band confirmed (hcooh R001, hcho_nh3 R006, R007), and a
band never found what the seed alone found except hcooh_h2o R010.

Cost on the 14 seeded reactions: seeds 2 min, their bands 34 min. Bands
on the 13 unseeded reactions (xTB found no saddle): 27 min. A
seed-first rule (bands only when the seed fails or there is no seed)
would have cost 2 + 6 + 27 = 35 min instead of 63 (44% less), and would
have kept a higher saddle in 2 of 17 reactions (about 4 kcal/mol in
energy). It also ties the ML saddles to xTB's mechanisms, which is what
the study compares, so it suits only engines too slow for D16.

### 7.8 MACE on a Kaggle GPU (benchmark, 2026-11-19)

MACE-OMol25 extra_large, mace-torch 0.3.17, one call at a time on two
saddles (hcooh R001 MACE, 5 atoms; cpd_ma R001 xTB, 20 atoms), median of
30 calls after 3 warm-up calls. Mac rows: 1 thread, other jobs running.

| Device | dtype | 5 atoms (ms) | 20 atoms (ms) |
|---|---|---|---|
| Mac CPU (the runs) | float64 | 277 | 1151 |
| Kaggle T4 GPU | float64 | 86 (3.2x) | 188 (6.1x) |
| Kaggle T4 GPU | float32 | 76 (3.6x) | 74 (15.6x) |
| Kaggle CPU, 1 thread | float64 | 349 | 1337 |

GPU float64 energies match CPU float64 to 0.001 meV; float32 moves them
by -0.08 and +0.42 meV. At 5 atoms float32 and float64 cost the same on
the GPU, so the call is launch-bound and the GPU mostly idle: several
processes can share it. Not adopted yet (runs left as they are); float32
would need a check on forces and frequencies before use. The GPU was
one T4 of a T4 x2 session (float64 at about 1/32 of its float32 rate,
which explains the 20-atom gap); a P100 should narrow it, untested.
Confirmed on a second run: model on cuda:0 (Tesla T4), 196 ms per
20-atom float64 call over 300 calls, peak GPU memory 1810 MB.

### 7.9 DFT cost per gradient (2026-11-19)

One ORCA 6.1.1 energy and gradient, wB97M-V/def2-TZVPD, 4 processes, on
xTB saddles, with two other refines running: hcooh R001 (5 atoms) 22 s,
hcho_nh3 R004 (8 atoms) 30 s. At the AIMNet2-NSE call counts (7.5,
53k / 129k / 101k counted calls for hcooh / hcooh_h2o / hcho_nh3) a full
D16 refinement in DFT would take about 13, 45 and 35 days on 4 cores. A
seed-only DFT attempt (median 262 real calls, 7.7) is about 1.5-2 h per
reaction.

### 7.10 Per-attempt time limit for xTB (PLAN D19, 2026-11-19)

The xTB cpd_ma refine had finished 13 of 38 reactions by 08:21 and then
spent 4.9 h on R014 (7.6). It was stopped at 13:20 without writing an
exit line, so the AIMNet2 and UMA cpd_ma runs kept waiting, and
restarted from a new source snapshot with `attempt_time_limit_s: 1200`.
The 13 finished reactions are kept; R014 onward runs under the limit.
Tests: `test_search_pair_attempt_time_limit` and related cases in
tests/test_ts_search.py (limit hit, alarm cleared, no limit, a stage
that catches Exception, first-status rule); 35 passed.

Result: the restarted run finished in 1675 s. R014 ended `time_limit`
after 20.0 min; the other 24 reactions took 8 min together. All 38:
confirmed 7, barrierless 4, endpoint_mismatch 8, wrong_imaginary_count
11, endpoint_changed 3, engine_error 4 (SCF), time_limit 1; the same
spread as 7.2. Diels-Alder at 298 K: endo (R038) 8.0, exo (R008) 10.3
kcal/mol, endo lower by 2.3 (literature 2.2).

AIMNet2-NSE on cpd_ma (seeded from xTB, 1 thread, 4531 s, no spin
warnings): reactions confirmed 11, wrong_imaginary_count 8,
endpoint_mismatch 7, endpoint_changed 6, saddle_not_converged 5,
barrierless 1; species ok 38, endpoint_changed 5, imaginary_minimum 1.
Diels-Alder at 298 K: endo 25.5, exo 26.6 kcal/mol (endo lower by 1.1;
xTB 8.0 / 10.3), dG_rxn -11.7 / -12.9 (xTB -35.6 / -35.7). To check in
the analysis: a bimolecular dG_act includes the entropy loss of
association (roughly +10 to +13 kcal/mol at 298 K), so xTB's 8.0 puts
the saddle's enthalpy near or below the reactants'.

UMA on cpd_ma (seeded from xTB, 1 thread, 18522 s): reactions confirmed
8, saddle_not_converged 9, wrong_imaginary_count 8, endpoint_mismatch 7,
endpoint_changed 4, barrierless 2; species ok 42, endpoint_changed 2.
Diels-Alder at 298 K: endo 23.2, exo 25.2 kcal/mol (endo lower by 2.0;
AIMNet2 25.5 / 26.6, xTB 8.0 / 10.3), dG_rxn -14.3 / -14.8. UMA and
AIMNet2 agree within 2.3 kcal/mol on both barriers; xTB sits 15-17
lower. xTB, AIMNet2-NSE and UMA are now complete on all four systems.

### 7.11 MACE on Kaggle, hcooh (2026-11-19)

Bundle 0f50206+dirty (built 07:06 UTC), T4 x2, float64, seeded from xTB,
full D16 search. hcooh: species 8 ok, reactions 4 of 4 confirmed, 3.24 h
(R001 40.0 min, R002 67.6, R003 80.1, R004 5.7). R001 repeats the Mac
CPU run (7964 against 7949 counted calls, dG_act 67.8 both); the Mac's
partial run is archived in ~/mse-archive-2026-11-18/
mace_hcooh_mac_cpu_partial/. Barriers at 1000 K (kcal/mol): R001 67.8,
R002 63.4, R003 27.2, R004 0.8, within 1.2 of UMA (68.3, 63.3, 26.3,
2.0); dehydration below decarboxylation by 4.4. Per counted call 0.305 s
with this job alone on its GPU, against 0.086 s for one call: the gap
of F14 appears without CPU sharing, so its cause sits in the search, not
in contention.

**F15: parallel jobs raced to download the model.** hcooh_h2o and
hcho_nh3 started at the same moment as hcooh and failed after 30 s with
"Model download failed and no local model found"; only hcooh got the
file. The runner still exited 0. Fix: the runner loads the engine once
before starting the jobs and exits non-zero if any system fails; the
rerun of the two systems downloads the model in a notebook cell first.

### 7.12 ORCA benchmark at the study's DFT level (2026-11-19)

`scripts/checks/orca_benchmark.py`, wB97M-V/def2-TZVPD TightSCF, ORCA
6.1.1, xTB saddles, idle Mac (4 performance + 6 efficiency cores),
results in data/benchmarks/orca_benchmark.csv. Wall seconds:

| Saddle | Atoms | SP, 4 proc | SP, 8 proc | EnGrad, 4 proc | EnGrad, 8 proc |
|---|---|---|---|---|---|
| hcooh R001 | 5 | 4.5 | 4.8 | 7.4 | 7.7 |
| hcho_nh3 R004 | 8 | 6.5 | 6.3 | 11.1 | 9.8 |
| hcooh_h2o R008 | 8 | 9.2 | 8.3 | 16.0 | 13.9 |
| cpd_ma R038 | 20 | 146 | 128 | 273 | 218 |

8 processes gain 0-20% over 4 (only 4 performance cores). Analytic
frequencies are not available for this functional: ORCA stops with "DFT-NL
dispersion correction analytic Hessian not yet implemented: Please use
!NUMFREQ instead". Frequencies therefore cost 6N gradients (5 atoms
about 4 min, 8 atoms 9-13 min, 20 atoms 7-9 h). The 7.9 timings (22 and
30 s per gradient) were taken with other jobs running and through ASE.

### 7.13 MACE on Kaggle, hcooh_h2o and hcho_nh3 (2026-11-20)

Rerun of the two F15 failures with the same bundle (0f50206+dirty, T4
x2, one system per GPU, float64, seeded from xTB, full D16 search),
11:00-20:41 UTC on 2026-11-19; merged on 2026-11-20.

| system | reactions | confirmed | other statuses | wall | per counted call |
| --- | --- | --- | --- | --- | --- |
| hcooh_h2o | 10 | 8 | 1 endpoint_mismatch, 1 wrong_imaginary_count | 9.69 h | 0.316 s |
| hcho_nh3 | 13 | 4 | 7 endpoint_mismatch, 1 saddle_not_converged, 1 wrong_imaginary_count | 8.67 h | 0.322 s |

Counted calls match UMA within 6 % on all three small systems (MACE
37990 / 109830 / 96367 against UMA 39065 / 111455 / 91296), and MACE
wall time is 3.6-5.0x UMA's. Per counted call stays at 0.31-0.32 s from
5 to 8 atoms.

Projection for cpd_ma (20 atoms, 38 reactions, UMA 240016 counted
calls, largest reaction 16162): 240k calls at 0.32 s (flat with size)
to 0.70 s (scaled by the 7.8 benchmark, 86 to 188 ms for 5 to 20 atoms)
gives 21-47 h on one T4. The runner gives one GPU per system, so one
11.5 h session covers a quarter to a half; the largest single reaction
(1.4-3.1 h) fits in one session, and resume skips finished reactions.

### 7.14 One system on both Kaggle GPUs (2026-11-21)

The runner gave one GPU per system, so cpd_ma alone left the second T4
idle. Added: `nrnet refine --reactions ID ... --skip-species`;
scripts/kaggle/plan_split.py (greedy split, heaviest reaction first, on
another engine's counted calls); make_bundle `--split-file` (stored as
splits/<system>.json); the runner runs each part in its own copy under
work-dir/parts/ and appends each part's new bytes to the main results
before packaging (CSV header kept once). Plan for cpd_ma from UMA calls
(data/cpd_ma/kaggle_split.json): 19 + 19 reactions, 119997 against
120019 calls. Check: AIMNet2 on hcooh, split 2 + 2 on the Mac CPU, 80 s;
the joined files equal the repo's run (same 4 statuses and routes, TS
energies equal to 0.000 meV, 8 species, 13 call rows, one header, 4 TS
frames). A resume from that zip skipped all and added no rows (2 s).
Expected cpd_ma wall on two T4s: 11-24 h, one or two sessions.

### 7.15 cuEquivariance kernels for MACE (2026-11-21)

The Kaggle MACE logs said "cuequivariance or cuequivariance_torch is not
available". Test on a Kaggle T4, MACE-OMol extra_large, float64, cpd_ma
reactant (20 atoms), 20 force calls on jittered geometries after one
warm-up call: 268 ms per call without, 129 ms with (2.07x); |dE| 4.5e-9
eV, max |dF| 3.6e-8 eV/A, so results stay comparable with the three
small systems run without it. Added `enable_cueq` to the mace block of
configs/engines.yaml (false by default) and make_bundle `--enable-cueq`
(sets it true in the bundle); the notebook installs cuequivariance,
cuequivariance-torch and cuequivariance-ops-torch-cu<CUDA major> first.
The cpd_ma bundle uses both the split (7.14) and the kernels.

### 7.16 ANI-1xnr added, Egret removed (PLAN D21, 2026-11-21)

ANI-1xnr loads through torchani 2.12 (`models.ANI1xnr`, 8-network
ensemble, BLYP/TZV2P, H C N O; weights `ani1xnr.pt`, sha256
beef541802e4cb3d...). The weights repository (atomistic-ml/ani-1xnr) has
no license file; torchani downloads them, the repository does not ship
them. Run in float64 for finite-difference frequencies. Cost on the Mac,
1 thread, after one warm-up call, 20 force calls on jittered geometries:
2.8 ms per call for HCHO (4 atoms), 4.5 ms for the cpd_ma endo adduct
(20 atoms), against about 1150 ms for MACE-OMol25 at 20 atoms (7.8).
New uv extra `ani` in its own environment; the lock gained 16 packages
and changed no existing pin. No spin input, so `closed_shell_only`:
reactions it can compute (no radical on either side, from the xTB
results) are hcho_nh3 10 of 13, hcooh 2 of 4, hcooh_h2o 7 of 10, cpd_ma
37 of 38, 56 of 65 in all. Egret-1 and Egret-1t removed from code,
settings, tests, Makefile and README; their hcooh results and the three
checkpoints moved to the archive outside the repository.

### 7.17 ANI-1xnr refine on all four systems (2026-11-21)

Seeded from xTB, 1 thread per system, all four in parallel on the Mac:
hcooh 202 s, hcho_nh3 720 s, hcooh_h2o 819 s, cpd_ma 1934 s (UMA took
5.1 h on cpd_ma). Confirmed saddles: hcooh 2/4, hcooh_h2o 4/10, hcho_nh3
4/13, cpd_ma 4/38; 14 of 65, against xTB 21, AIMNet2 29 and UMA 22 (MACE
cpd_ma still on Kaggle). Of the 56 reactions it can compute, 42 end
without a saddle; on cpd_ma 12 wrong_imaginary_count, 10
endpoint_mismatch, 5 endpoint_changed, 5 barrierless. Neither
Diels-Alder saddle is confirmed (R008 exo, R038 endo: endpoint_mismatch;
UMA 25.2 / 23.2, AIMNet2 26.6 / 25.5 kcal/mol). Hydroxycarbene
(CH2O-63452dc2) is not a minimum on ANI-1xnr (-208 cm^-1), so hcho_nh3
R009 is confirmed but has no barrier; cpd_ma has 4 species
endpoint_changed and 2 imaginary_minimum.

Forward barriers where ANI-1xnr confirmed (kcal/mol; xTB, AIMNet2, UMA,
MACE, ANI-1xnr; 1000 K for hcooh and hcooh_h2o, 298.15 K otherwise):

| Reaction | xTB | AIMNet2 | UMA | MACE | ANI-1xnr |
|---|---|---|---|---|---|
| HCOOH -> CO2 + H2 (hcooh R001) | 64.9 | 79.5 | 68.3 | 67.8 | 60.2 |
| HCOOH -> CO + H2O (hcooh R002) | 74.7 | 79.1 | 63.3 | 63.4 | 45.5 |
| HCOOH + H2O -> HC(OH)3 (hcooh_h2o R008) | 67.4 | 83.1 | 76.5 | 76.7 | 83.9 |
| H2CO3 -> CO2 + H2O (hcooh_h2o R010) | 37.2 | 42.8 | 39.6 | 39.3 | 32.6 |
| HCHO + NH3 -> aminomethanol (hcho_nh3 R004) | 26.4 | 43.4 | 39.2 | 39.4 | 55.7 |
| HCHO -> CO + H2 (hcho_nh3 R006) | 68.1 | 103.2 | 81.5 | 81.9 | 72.6 |
| aminomethanol dehydration (hcho_nh3 R013) | 60.8 | 58.5 | 55.1 | 55.0 | 41.0 |
| vinylcyclopentadiene H shift (cpd_ma R027) | 27.1 | 36.1 | 26.6 | - | 30.4 |

ANI-1xnr departs from the mean of UMA and MACE by -18 to +16 kcal/mol,
in both directions, so this is not a uniform shift. The CO-forming and
dehydration steps come out lowest, as a GGA (BLYP) reference would
predict; the HCHO + NH3 addition comes out 16 kcal/mol higher. UMA and
MACE agree within 0.5 kcal/mol on every reaction they share. Summary
CSVs: data/<system>/checks/refine_ani1xnr.csv (and refine_mace.csv for
hcooh_h2o and hcho_nh3, written now).

### 7.18 Duplicates within each system (PLAN D22, 2026-11-21)

Read from stored results only. MD events to reactions, merged by `freeze`
on stereo-aware reacting molecules: hcooh 98 -> 4, hcooh_h2o 56 -> 10,
hcho_nh3 20 -> 13, cpd_ma 56 -> 38 (230 -> 65). Within a system, no two
reactions share both sides in either direction, so the undirected count
equals the row count in every system. `refine` keeps one saddle geometry
per reaction (`ts.extxyz`) and one geometry per species key, so nothing
is left to merge. Confirmed saddles kept as energies only, more than
0.1 kcal/mol from the kept saddle (xtb, aimnet, uma, mace, ani1xnr):
hcooh 1, 1, 1, 1, 1; hcooh_h2o 1, 2, 1, 1, 1; hcho_nh3 0, 1, 0, 0, 2;
cpd_ma 0, 0, 0, -, 0; 14 in all. Across systems, hcooh's four reactions
all reappear in hcooh_h2o (hcooh R001-R004 = hcooh_h2o R001, R003, R005,
R007); by D22 they stay in both.

Energy-only saddles by reaction, energy relative to the kept saddle in
kcal/mol (seed = search started from xTB's saddle; pair N = the engine's
own band on the N-th MD pair). The kept saddle is the lowest G at the
first ranking temperature (D16), so a saddle lower in E can lose on G.

| System | Engine | Reaction | Kept from | Other saddle |
|---|---|---|---|---|
| hcooh | xtb | R001 HCOOH -> CO2 + H2 | pair 1 (= pair 2) | pair 0: +43.6 |
| hcooh | aimnet | R002 HCOOH -> CO + H2O | seed | pair 1: -2.4 |
| hcooh | uma | R003 CHO2 -> CO2 + H | seed (= pair 1) | pairs 0, 2: +17.9 |
| hcooh | mace | R003 CHO2 -> CO2 + H | pair 1 (= seed) | pairs 0, 2: +15.8 |
| hcooh | ani1xnr | R001 HCOOH -> CO2 + H2 | pair 2 | seed, pair 1: -2.9 |
| hcooh_h2o | xtb | R001 HCOOH -> CO2 + H2 | pair 2 | pairs 0, 1: +43.6 |
| hcooh_h2o | aimnet | R001 HCOOH -> CO2 + H2 | pair 2 (= pairs 0, 1) | seed: -4.1 |
| hcooh_h2o | aimnet | R003 HCOOH -> CO + H2O | seed | pairs 0-2: -2.4 |
| hcooh_h2o | uma | R005 CHO2 -> CO2 + H | pair 1 (= seed, pair 2) | pair 0: +17.9 |
| hcooh_h2o | mace | R005 CHO2 -> CO2 + H | seed (= pairs 1, 2) | pair 0: +15.8 |
| hcooh_h2o | ani1xnr | R008 HCOOH + H2O -> HC(OH)3 | pair 0 (= seed) | pair 1: -2.9 |
| hcho_nh3 | aimnet | R013 aminomethanol dehydration | pair 0 | seed: +4.2 |
| hcho_nh3 | ani1xnr | R004 HCHO + NH3 -> aminomethanol | seed | pair 2: +14.2 |
| hcho_nh3 | ani1xnr | R006 HCHO -> CO + H2 | pair 2 | pair 0: +11.3 |

The same alternative appears in hcooh and hcooh_h2o for the same engine
(xtb +43.6, uma +17.9, mace +15.8, aimnet -2.4), so these are real
saddles, not noise.
