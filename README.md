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
  validate.py      the milestone battery
run_milestone.py   prints the battery
```

## Next (not yet built)

The evolutionary layer: an evolvable controller (compact recurrent net) as the
heritable genome, mutated at the thermodynamically-forced rate `1−R_n`; a finite
shared gradient so selection is real; reproduction as the branching step. Then run
it and watch what evolves — whether predictive machinery arises, how its complexity
scales with environmental predictability, and where the error-catastrophe ceiling
on sustainable intelligence sits.
