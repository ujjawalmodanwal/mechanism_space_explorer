# Mechanism Space Explorer: status report

21 November 2026. Decisions are in `PLAN.md` (D1-D22); every fix and
check, with its numbers, is in `docs/discovery-log.md`.

## Summary

The question: if a nanoreactor study swaps density functional theory for
a machine-learned potential, does it predict different products, or a
different pathway to them?

What is built: one pipeline, end to end. xTB metadynamics discovers
reactions once; the reactions are frozen to disk; then every energy
engine re-optimizes the same molecules, searches the same transition
states and computes free energies, so any difference between the
resulting networks comes from the engine alone. Four reacting systems,
five engines (GFN2-xTB, MACE-OMol25, UMA-S 1.2.1, AIMNet2-NSE,
ANI-1xnr), each system reported at its own temperatures.

Status: xTB, AIMNet2-NSE, UMA and ANI-1xnr have refined all four
systems; MACE-OMol25 has refined three and is finishing the fourth on a
GPU. The DFT reference (ωB97M-V/def2-TZVPD) is on hold: a full DFT
refinement would take about five weeks on the available machine, so
accuracy is judged against gas-phase reference tables and published
barriers.

Findings so far:

1. **The engine changes the chemistry.** xTB makes HCOOH
   decarboxylation the faster channel; the OMol25 models make
   dehydration faster by 4-5 kcal/mol, as the literature reports. xTB
   puts every CO-forming reaction free energy 39-46 kcal/mol above the
   reference tables; AIMNet2, UMA and MACE land within 0.1-3.5
   kcal/mol.
2. **Training data matters more than architecture.** UMA and MACE, two
   different architectures trained on the same OMol25 data, agree
   within 0.5 kcal/mol on every barrier they share.
3. **Selectivity survives, absolute barriers do not.** Every engine that
   found both Diels-Alder saddles favours endo by 1.1-2.3 kcal/mol
   (reported: 2.2), but the endo barrier itself is 8.0 kcal/mol with
   xTB and 23-26 kcal/mol with the ML engines.
4. **Reaction training alone does not help here.** ANI-1xnr, trained on
   nanoreactor data at BLYP, confirmed the fewest transition states, no
   Diels-Alder saddle, and barriers 18 kcal/mol below to 16 above the
   OMol25 pair.
5. **The discovery and filtering stages lose information,** measured in
   section 6. A redesign of event detection and transition-state
   finding is planned (section 7).

## 1. Systems, engines, temperatures

| System | Atoms | Start | Reported at |
|---|---|---|---|
| hcooh | 5 | formic acid, trans and cis | 1000 K |
| hcooh_h2o | 8 | formic acid + water | 1000 K |
| hcho_nh3 | 8 | formaldehyde + ammonia | 298.15 K, 600 K |
| cpd_ma | 20 | cyclopentadiene + maleic anhydride, endo and exo | 298.15 K, 493.15 K |

| Engine | Model | Trained on | Charge and spin |
|---|---|---|---|
| GFN2-xTB | tblite 0.8.0 | semi-empirical method | yes |
| MACE-OMol25 | extra-large | OMol25, ωB97M-V/def2-TZVPD | yes |
| UMA | uma-s-1p2p1, OMol25 task | OMol25 and other sets | yes |
| AIMNet2-NSE | aimnet2-nse_0 | ωB97M | yes |
| ANI-1xnr | torchani 2.12 | nanoreactor data, BLYP/TZV2P | no (closed-shell only) |
| DFT reference | ORCA 6.1.1 | ωB97M-V/def2-TZVPD | on hold |

Checkpoints, checksums and licences are in `README.md`. Why other
potentials (NequIP, Allegro, SpookyNet, CHGNet, M3GNet, ANI-1x/2x,
Egret) are not used: `PLAN.md`, D21.

## 2. The pipeline as built

1. **Discovery.** GFN2-xTB RMSD metadynamics (xtb 6.7.1) inside a
   spherical wall, 300 K bath, 0.5 fs step, frames every 5 fs, 10 ps per
   run, settings after Zhang et al. (JCTC 2023) and Grimme (JCTC 2019).
   Production: 600 runs per small system (two push strengths, six groups
   of 50), 350 for cpd_ma.
2. **Events.** Bonds are read from distances (1.2 x covalent radii). A
   molecule counts as a state if it lives at least 50 fs; an event is a
   change between two such states.
3. **Freeze.** Only the atoms whose bonds change are kept. Events with
   two or more radicals on one side are rejected (open-shell singlets
   and triplets are out of scope). Both sides are relaxed with xTB and
   must keep their identity. Events with the same reacting molecules,
   stereochemistry included, merge into one reaction; the first three
   become its starting pairs. Known literature reactions the MD missed
   are added by hand and flagged.
4. **Refine, per engine.** Every molecule is optimized and its
   frequencies computed. Every reaction is searched from each starting
   pair (13-image path, plain then climbing; ML engines also start once
   from xTB's saddle), refined with Sella, and accepted only with
   exactly one imaginary mode below -50 cm^-1 and both ends relaxing
   back to the expected molecules. The confirmed saddle with the lowest
   free energy is kept.
5. **Network.** One graph per engine and system, nodes the state of the
   whole box. A step with no barrier keeps its arrow with
   G(TS) = max(G_A, G_B).

## 3. Discovery

| System | MD events | Rejected: two radicals | Rejected: changed on relaxing | Reactions | Known in literature |
|---|---|---|---|---|---|
| hcooh | 652 | 554 | 0 | 4 | decarboxylation (found by the MD) |
| hcooh_h2o | 184 | 128 | 0 | 10 | water-assisted decarboxylation (found) |
| hcho_nh3 | 173 | 153 | 0 | 13 | aminomethanol dehydration (added by hand) |
| cpd_ma | 111 | 4 | 58 | 38 | Diels-Alder exo (found), endo (added by hand) |

The main channels are reproducible: 100 independent check runs find
most reactions the 200 main runs saw twice or more (hcho_nh3 2 of 2,
hcooh 2 of 3, hcooh_h2o 4 of 5; hcooh decarboxylation 44 and 24 events,
dehydration 17 and 8). Rare reactions are not converged.

## 4. Results

### 4.1 Coverage: confirmed transition states

| System (reactions) | xTB | AIMNet2-NSE | UMA | MACE | ANI-1xnr |
|---|---|---|---|---|---|
| hcooh (4) | 4 | 4 | 4 | 4 | 2 |
| hcooh_h2o (10) | 6 | 8 | 6 | 8 | 4 |
| hcho_nh3 (13) | 4 | 6 | 4 | 4 | 4 |
| cpd_ma (38) | 7 | 11 | 8 | running | 4 |

ANI-1xnr cannot treat radicals, so 2, 3, 3 and 1 reactions of these
systems are outside its reach.

### 4.2 Reaction free energies against reference tables (kcal/mol)

Tables: 298 K enthalpies of formation and entropies (NIST/JANAF); heat
capacity changes ignored.

| Reaction | T (K) | Tables | xTB | AIMNet2 | UMA | MACE | ANI-1xnr |
|---|---|---|---|---|---|---|---|
| HCOOH -> CO2 + H2 | 1000 | -26.5 | -38.2 | -29.5 | -30.0 | -29.9 | -45.6 |
| HCOOH -> CO + H2O | 1000 | -26.7 | +19.5 | -27.8 | -27.0 | -27.1 | -36.0 |
| HCHO -> CO + H2 | 298 | -8.3 | +31.0 | -5.9 | -7.7 | -7.6 | -1.4 |
| HCHO -> CO + H2 | 600 | -16.2 | +22.6 | -14.4 | -16.1 | -16.0 | -9.8 |

### 4.3 Barriers (forward free energy of activation, kcal/mol)

Measured from the separated reacting molecules at 1 atm.

| Reaction | T (K) | xTB | AIMNet2 | UMA | MACE | ANI-1xnr | Literature |
|---|---|---|---|---|---|---|---|
| HCOOH -> CO2 + H2 | 1000 | 64.9 | 79.5 | 68.3 | 67.8 | 60.2 | dehydration dominant in the gas phase |
| HCOOH -> CO + H2O | 1000 | 74.7 | 79.1 | 63.3 | 63.4 | 45.5 | |
| HCHO + NH3 -> aminomethanol | 298 | 26.4 | 43.4 | 39.2 | 39.4 | 55.7 | about 34; 31.4 (B3LYP-D3(BJ), Zhang 2023) |
| aminomethanol -> CH2=NH + H2O | 298 | 60.8 | 58.5 | 55.1 | 55.0 | 41.0 | 55.0-55.9 (Feldmann 2005) |
| HCHO -> CO + H2 | 298 | 68.1 | 103.2 | 81.5 | 81.9 | 72.6 | |
| Diels-Alder endo | 298 | 8.0 | 25.5 | 23.2 | running | none found | endo kinetic |
| Diels-Alder exo | 298 | 10.3 | 26.6 | 25.2 | running | none found | exo minus endo 2.2 |

- HCOOH: dehydration minus decarboxylation is +9.8 (xTB), -0.4
  (AIMNet2), -5.0 (UMA), -4.4 (MACE), -14.7 (ANI-1xnr). Only xTB
  reverses the reported order.
- Water-assisted HCOOH steps were confirmed only by MACE
  (decarboxylation 79.8, dehydration 78.3) and AIMNet2 (dehydration
  106.2). At 1000 K and 1 atm they come out above the bare steps,
  because bringing in the water costs entropy; the reported lowering
  (up to about 16 kcal/mol, Wolf, Turney, Schaefer 2020) is in
  electronic energy, so a like-for-like comparison is still to do.

### 4.4 Engine agreement and alternative saddles

- UMA and MACE agree within 0.5 kcal/mol on every shared barrier.
- ANI-1xnr departs from the UMA-MACE mean by -18 to +16 kcal/mol, in
  both directions; formaldehyde's hydroxycarbene isomer is not a minimum
  on its surface.
- Some reactions have a second confirmed saddle 2-44 kcal/mol from the
  kept one; the same alternatives appear in hcooh and hcooh_h2o for the
  same engine, so they are real (log 7.18).

### 4.5 Cost

| Engine | Force call (20 atoms) | Refinement wall time |
|---|---|---|
| ANI-1xnr | 4.5 ms, Mac CPU | 3-32 min per system |
| UMA | about 10x faster than MACE, Mac CPU | cpd_ma 5.1 h |
| MACE-OMol25 | 1150 ms Mac CPU; 268 ms T4 GPU, 129 ms with cuEquivariance | 3.2-9.7 h per small system on a T4 |
| DFT, ωB97M-V/def2-TZVPD | gradient 3.6-4.5 min, 4 cores | frequencies 7-9 h per 20-atom structure |

## 5. Faults found and fixed

| | Fault | Fix |
|---|---|---|
| F1 | xtb reads the wall radius in bohr; a 4 A wall was 2.12 A | conversion; all data regenerated |
| F2 | `xtb --opt` crashes in some builds | other optimizer engine |
| F3 | every group started from the same equilibration frames | seeded draw per group |
| F4 | groups compared atom-numbered SMILES | compare numbering-free keys |
| F5 | H-H cutoff below xTB's H2 bond: H2 read as two atoms | H radius 0.37 A |
| F6 | xtb writes energy 0 after a failed MD step | reading stops there |
| F7 | endpoints taken at the crossing relax to the other side | frames taken away from the crossing |
| F8 | a literature pair replaced an MD pair | MD-found reactions only get a label |
| F9 | Sella's internal-coordinate step fails | recorded as not converged |
| F10 | tblite's 250 SCF cycles too few on stretched shapes | 1000 cycles |
| F11 | thread count changes borderline outcomes | one thread per job |
| F12 | an assisted reaction matched by whole states | match on reacting molecules (D17) |
| F13 | AIMNet2 wB97M-D3 ignores spin | spin-aware AIMNet2-NSE (D18) |
| F14 | force-call counter counts optimizer steps | corrected in the cost analysis |
| F15 | parallel GPU jobs raced to download the model | download first |

## 6. Limitations: where information is lost

These are measured, not guessed; most need recomputation to fix.

1. **Event detection.** A single distance cutoff makes bonds flicker,
   and the 50 fs rule then removes noise and real chemistry together.
   On 100 hcho_nh3 runs: 42,351 raw state changes become 25 events.
   With separate forming and breaking distances (1.15 and 1.5 x
   covalent radii) the same rule keeps 35 events, and 18 distinct
   states shorter than 50 fs remain that the rule deletes.
2. **Two-radical events are rejected** at freeze (section 3), which
   removes bond breaking into two radicals and radical recombination.
3. **Only the first three events per reaction are searched**: 88 events
   unsearched in hcooh, 34 in hcooh_h2o, 4 in hcho_nh3, 10 in cpd_ma.
4. **Saddles that lead elsewhere are discarded.** A saddle whose ends
   are not the expected molecules (often a two-step event) is dropped
   without storing where it leads. UMA: 3 of 10 reactions in
   hcooh_h2o, 8 of 13 in hcho_nh3, 7 of 38 in cpd_ma.
5. **Failed searches are not retried** with a second method; on cpd_ma
   18-21 reactions per engine end this way.
6. **Confirmation relaxes from the saddle,** a cheap stand-in for an
   intrinsic reaction coordinate.
7. **Only the kept saddle's geometry is stored,** chosen at the first
   reporting temperature; other saddles keep only their energy.
8. **Thermochemistry:** one conformer per molecule, rotational symmetry
   number 1, no tunnelling correction, 1 atm standard state.
9. **Literature references** are partly still to be checked against
   their primary sources.
10. **Not yet run on the final data:** the network rankings and the
    engine-by-engine evaluation.

## 7. Next steps

1. Merge MACE-OMol25 on cpd_ma.
2. Redesign event detection on the saved trajectories (no new MD):
   bonds with separate forming and breaking distances or a hidden
   Markov model (as in Zhang et al. 2023), thresholds set from the
   trajectories' own distance distributions, and every event labelled
   instead of dropped.
3. Find transition states from the frames where bonds change (as in
   AutoMeKin): optimize to a saddle, check the frequencies, follow the
   intrinsic reaction coordinate both ways, keep the path search as a
   fallback, and store every saddle with its frequencies so the best
   barrier can be chosen at every temperature.
4. Run two studies on that base: the discovery network (does the
   network change with the engine?) and a benchmark on literature
   reactions with high-level references.
5. Revisit the engines, including the reaction-trained AIMNet2-rxn.
6. Decide the DFT validation: single points on ML geometries, gas-phase
   reference tables and coupled-cluster literature barriers.

## 8. Reproducing

Setup and commands: `README.md`. Frozen reactions:
`candidates/<system>/`. Refinement results:
`results/refine/<system>/<engine>/`. Per-reaction summaries:
`data/<system>/checks/refine_<engine>.csv`
(`scripts/checks/refine_summary.py`). The MD trajectories (4.6 GB) are
not in the repository; `nrnet discover` and `nrnet sweep` regenerate
them.

## References

- L.-P. Wang et al., "Discovering chemistry with an ab initio
  nanoreactor", *Nat. Chem.* 6, 1044 (2014).
- L.-P. Wang, R. T. McGibbon, V. S. Pande, T. J. Martinez, "Automated
  discovery and refinement of reactive molecular dynamics pathways",
  *J. Chem. Theory Comput.* 12, 638 (2016).
- S. Grimme, *J. Chem. Theory Comput.* 15, 2847 (2019).
- Y. Zhang et al., "Automated exploration of reaction networks and
  mechanisms based on metadynamics nanoreactor simulations", *J. Chem.
  Theory Comput.* 19, 8718 (2023).
- J. Zeng et al., "ReacNetGenerator", *Phys. Chem. Chem. Phys.* 22, 683
  (2020).
- M. Döntgen et al., "Automated discovery of reaction pathways, rate
  constants, and transition states using reactive molecular dynamics
  simulations", *J. Chem. Theory Comput.* (2015).
- E. Martínez-Núñez et al., "AutoMeKin2021: an open-source program for
  automated reaction discovery", arXiv:2105.03078.
- M. M. Wolf, J. M. Turney, H. F. Schaefer, "High level ab initio
  investigation of the catalytic effect of water on formic acid
  decomposition and isomerization", *Phys. Chem. Chem. Phys.* (2020).
- A. Feldmann et al., "Aminomethanol water elimination: theoretical
  examination", *J. Chem. Phys.* (2005).
- J. Marks, J. Vandezande, J. Gomes, "Reliable and efficient automated
  transition-state searches with machine-learned interatomic
  potentials", arXiv:2604.00405 (2026).
- Models: see `README.md`, Citing.
