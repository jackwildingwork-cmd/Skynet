# thermoevo — evolution from thermodynamic first principles

A ground-up substrate built on the **Persistence Theorem** (Wilding, *The
persistence theorem: necessary and sufficient conditions for physical
persistence*). The goal is narrow and specific: **derive evolution from
thermodynamics** — variation forced by the second law, selection forced by a
finite gradient — and then see *what evolves* when the environment rewards
prediction, i.e. when carrying an internal model of the world captures more free
energy.

This replaces an earlier economic-analogy simulation, which was discarded: it was
a loose metaphor, whereas the paper gives exact physics.

## The theory it stands on

The paper reduces persistence to one self-referential loop, the **coupling chain**:

```
N_s --eta--> E_h --defect kinetics--> {ΔV_j} --min_j--> Ω --TST--> R_n --branching--> N_s(t+1)
```

and three conditions, each a precise inequality on a shared variable:

- **CI — gradient coupling with surplus.** Structural stock `N_s` obeys a Langevin
  equation with additive noise and an absorbing boundary at `N_s=0`. CI is
  `⟨dN_s/dt + γ_n N_s⟩ > 0` (positive net drift). The first-passage theorem: drift
  >0 → survival approaches a nonzero constant; ≤0 → extinction with probability 1.
- **CII — active homeostasis.** Integrity `Ω = min_j ΔV_j/ΔV_max` is the shallowest
  Kramers well; thermal fluctuation escapes it (structural death) at rate
  `ω₀·exp(−ΔV/k_BT)`. Maintenance spends the process's own energy `E_h` to hold the
  wells deep. Passive stability is dislodged with probability 1 (Kramers). Fidelity
  follows by transition-state theory: `R_n(Ω) = 1 − exp(−Ω·ΔE/k_BT)`.
- **CIII — loop closure.** `R_0 = R_n·φ`. Below `R_0=1` the branching-process
  extinction theorem gives certain extinction; above it, extinction probability
  `q<1` and the class persists forever with probability `1−q`.

**Variation is not an operator we add** — it is forced: `R_n<1` at any `T>0`.
**Selection is not a fitness function we impose** — it is the finite gradient
driving less-effective couplers to the absorbing boundary faster.

## Why "indefinite persistence" is testable in finite time

You cannot run forever. You do not need to. The theory maps the `T→∞` question
onto two **finite-time-measurable** quantities via proven theorems: the drift sign
of `N_s` (CI first-passage) and the offspring distribution (CIII branching → `q`).
Measure those with statistical confidence, then invoke the theorems. `q<1` is the
evidence for indefinite class-level persistence.

## How the environment rewards intelligence

The gradient carries a hidden type `θ(t)` (an AR(1) process; correlation `ρ` is the
**predictability** knob) that an agent must match its coupling to. The agent commits
its action using only past cues, so an agent that *models* `θ` pre-positions and
captures more. This is the thermodynamics-of-prediction result made concrete: the
extra free energy a predictive agent captures is bounded by the **predictive
information** its internal state holds about the future gradient — available only to
the extent the environment is predictable. A model has upkeep cost, so intelligence
evolves only when the environment is predictable enough to pay for it.

## Milestone 1 (this commit): the core, validated

Before building the evolutionary layer, the substrate is checked against the
paper's theorems. `python run_milestone.py`:

| check | result |
|---|---|
| **CI** first-passage | μ<0 → survival 0; μ=0 → `t^-1/2`; μ>0 → nonzero plateau matching `1−exp(−2μb/σ²)` |
| **CII** Kramers + Gompertz | passive stability → survival 0; active → 0.99; **Gompertz–Makeham emerges** (log-hazard linear in age, slope = c/k_BT, predicted 0.040 vs fitted 0.040) |
| **CIII** branching | `q_MC = q_analytic` across R₀=0.6→2.4; **error catastrophe** at the Eigen threshold `R_n=1/φ` |
| **prediction reward** | value of an internal model ≈0 when white (ρ=0), rising to +0.25 at ρ=0.98, tracking predictive information (0→1.09 nats) |

All four pass; 10 unit tests in `tests/test_thermo.py` gate the core.

```
thermoevo/
  constants.py     dimensionless thermodynamic parameters (k_B=1; T is the knob)
  langevin.py      CI: N_s dynamics + drift-diffusion first-passage (exact + MC)
  integrity.py     CII: Kramers wells, Ω, R_n(Ω), escape/mortality
  branching.py     CIII: extinction probability q (analytic + MC), error catastrophe
  environment.py   structured predictable gradient (θ AR(1) + noisy cue)
  controller.py    fixed reactive vs predictive (Kalman) controllers; predictive info
  validate.py      the milestone-1 battery
  genome.py        evolvable recurrent-controller genome (well-mixed model)
  evolve.py        milestone-2 engine (well-mixed shared gradient, branching)
  evo_experiment.py  the milestone-2 sweeps
  neuralnet.py     general recurrent controller (vector I/O) for the spatial world
  field.py         discrete, patchy, capped, replenishing resource field (m3)
  producers.py     autotrophs: couple a universal gradient (sunlight) -> vegetation
  world.py         spatial foraging: herbivores graze producer biomass (m3/m4)
  world_experiment.py  the milestone-3 experiment
  trophic_experiment.py  the milestone-4 two-trophic-level experiment
run_milestone.py   prints the milestone-1 battery
```

## Milestone 2: evolution on the substrate

The evolutionary layer (`genome.py`, `evolve.py`): a population of compact
recurrent-controller genomes on a finite shared gradient. Variation is **forced,
not coded** — offspring weights are perturbed with magnitude `mut·μ`,
`μ = exp(−Ω·ΔE/k_BT)` (the paper's VAR rate), so temperature and integrity set the
mutation rate. Selection is **not imposed** — it is the finite gradient split by
match quality. Reproduction is the branching step. `python -m thermoevo.evo_experiment`:

**Persistence is reproduction-carried.** Populations run tens of generations of
turnover (founders replaced), the class persisting while every individual dies —
exactly the paper's class-vs-instance distinction, now with intrinsic Gompertz
mortality emerging from the Ω dynamics.

**Prediction evolves only when the environment affords it.** Predictive
information the evolved controllers carry about the gradient, vs predictability ρ:

| ρ | 0.00 | 0.30 | 0.60 | 0.90 | 0.98 |
|---|---|---|---|---|---|
| evolved predictive info (nats) | 0.000 | 0.019 | 0.087 | 0.269 | 0.412 |
| evolved coupling skill | 0.309 | 0.342 | 0.377 | 0.449 | 0.488 |

Zero when the gradient is white (nothing to predict), rising monotonically with
predictability. Intelligence is rewarded in proportion to how predictable the
world is — now as an *evolved* outcome, not a designed one.

**There is a thermodynamic evolvability window.** Because variation is forced by
temperature, adaptation (selection improving coupling over generations) happens
only in an intermediate `μ` band:

| ΔE (→ μ at Ω=1) | 3.0 (0.050) | 4.0 (0.018) | 5.0 (0.007) | 6.0 (0.002) |
|---|---|---|---|---|
| skill change over generations | −0.005 | **+0.023** | **+0.032** | +0.009 |
| regime | mutation load | adapts | adapts | frozen |

Too much forced variation degrades the genome (Muller's ratchet / error-catastrophe
side); too little freezes evolution. The Goldilocks band is a direct consequence of
the mutation–selection–drift balance the theory predicts.

**Honest limits.** These are directional, drift/load-limited effect sizes (small
populations, forced variation, pure-mutation controller evolution) — the ρ sweep
is clean and monotone; the skill deltas are modest but consistent. And there is no
hard extinction-catastrophe here, because the proportional gradient always feeds
someone: high `μ` degrades skill rather than collapsing the population. A true
error-catastrophe-to-extinction would need coupling to be survival-critical
(absolute rather than proportional capture) — a deliberate next step, not yet built.

## Milestone 3: foraging a discrete, replenishing gradient world

The well-mixed gradient of milestones 1–2 is replaced by an environment that
better reflects a real one (`field.py`, `world.py`, `world_experiment.py`):
**discrete fertile patches** on a torus that **regrow to a fixed capped total**
(not scaled to the population), which agents must **find**. Condition I is now
explicit and local — intake must exceed the cost of *finding* (moving) and
consuming, or structural stock drifts to the absorbing boundary. Agents sense
their local resource gradient and hunger; their evolved recurrent controller
chooses a move; they compete for patches that deplete when eaten. Selection acts
on nothing but net-positive foraging. `python -m thermoevo.world_experiment`:

**Foraging intelligence evolves.** Chemotaxis alignment — how well an agent's
movement climbs the sensed resource gradient — rises from the random-genome
baseline **0.13 → 0.45** (×3.6) over ~12–18 generations, while the population
self-regulates at a carrying capacity set by the field (not by the population),
and every individual dies while the class persists.

**The value of that intelligence depends on the field's structure.** Sweeping
patch density (evolved chemotaxis):

| patches | 30 | 60 | 120 | 240 |
|---|---|---|---|---|
| evolved chemotaxis | 0.16 | 0.42 | 0.45 | 0.49 |
| carrying capacity | 36 | 42 | 80 | 152 |

**Denser** patches evolve *more* gradient-climbing — a denser field is spatially
continuous and followable, so chemotaxis pays. Sparse isolated patches leave gaps
where climbing doesn't help and finding falls back on exploration and luck; the
population is smaller but still persists (no extinction reached in this range).
This was the opposite of the initial guess and is reported as found.

Competition (patch depletion, density-dependent capacity) and luck (stochastic
encounters, offspring scatter) are active in this substrate; cooperation is not
built in and did not appear. Effect sizes here are large and clean — the spatial
"finding" structure gives selection a much stronger handle than the well-mixed
model did.

## Milestone 4: two trophic levels in one tier

The environment gets bigger. Until now the gradient the foragers ate was *given* —
an abstract logistic field. Now it is **produced**. A population of **autotroph
processes** (`producers.py`) couples to a **universal gradient** — sunlight,
available to every producer at every cell — and fixes it into **standing structural
stock**: the vegetation. The old foragers become **herbivores** (`world.py`): they
cannot touch sunlight, so their gradient is the producers' biomass, which is patchy
and mobile and must be **found**. Both levels are the same validated core (N_s
Langevin + absorbing boundary, Ω Kramers wells, forced-error reproduction); the
environment, not the code, makes them differ. `python -m thermoevo.trophic_experiment`:

**A universal gradient still has a finite carrying capacity.** Sunlight is
everywhere, so there is no finding problem — yet producers do not blow up. Two
brakes bound them: **self-shading** (neighbours split the light, a Beer–Lambert
share `b_i/(k+B_local)`) and **recruitment limitation** (a seed establishes only on
a safe, un-shaded site). Together they turn an unbounded gradient into a stable
standing crop (crop CV over the last 200 ticks ≈ 0.00). "Unlocking is not
accessing": the flux is universal, the *accessible* flux per coupler is finite.

**The two levels coexist — top-down and bottom-up at once.** Add herbivores and
neither collapses. Grazing holds producers **~66% below** their ungrazed crop
(top-down control), while herbivores settle at a carrying capacity set by producer
productivity (bottom-up, ~900 on the default field), every individual dying while
both classes persist. A stable ecology, not a boom–bust cycle, across seeds.

**Intelligence is selected by the gradient's structure, not the organism.** This is
the sharp result. The *same* process, facing a *universal* gradient (producers),
evolves no foraging skill — there is nothing to climb. Facing a *patchy* gradient
(herbivores), it evolves gradient-climbing chemotaxis from a random baseline
**~0.03 → ~0.62**. A mind pays exactly when the free energy must be found:

| | producers | herbivores |
|---|---|---|
| gradient | sunlight (universal) | producer biomass (patchy) |
| finding problem | none | must locate clumps |
| bound by | self-shading + safe sites | food supply (grazing) |
| evolved foraging χ | — (n/a) | 0.03 → 0.62 |

This is RGC's "structural stock **is** a gradient for the next consumer", realised
inside a single tier: producers → consumers. `tests/test_thermo.py` gates it
(producers persist on a universal gradient; the two levels coexist with the
producers grazed materially below their ungrazed crop). 16 tests pass.

## Next

The environment is now a stable two-level ecology, which is the substrate the
tier transition needs. Deferred and next: **n−1 / higher-level individuality** —
let a cooperating family of tier-1 processors satisfy CI/CII/CIII against a *new*
gradient class and become a single tier-2 processor (multicellularity), using the
RGC dual criterion (persistence `S ≥ S*` and coordination efficiency `χ ≥ χ*`).
Also open: producer-side evolution of defense (a real coevolutionary arms race),
survival-critical coupling to surface a true extinction-catastrophe, and letting
controller topology evolve so complexity can deepen.
