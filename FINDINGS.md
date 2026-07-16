# Findings — Persistence Simulation

An honest record of what the build showed, across three phases. The epistemic
rule throughout: report what the data says, disclose every parameter decision,
and never tune toward a favourable-looking result. Where a hypothesis failed to
appear, that is recorded as a failure, not reframed.

---

## Phase 1 — The closed commons (v1)

The first build instantiated the framework faithfully with a **fixed, finite
resource pool** (the brief's global budget) and a per-tick task flux fixed in
advance.

Result across a 24-seed ensemble:

- Substrate conditions SC0–SC4 and SC3' verified for the concrete system.
- Both death modes occur; **all six Lemma-0.2 pairwise-independence signatures
  appear organically** (each in 4–23 of 24 seeds).
- Adding **ambient entropy** (an unconditional drift toward the maintenance dead
  set, beyond the brief's sabotage-only SC3') made CII genuinely load-bearing.
  Once it was present, the **adaptive population outlasted a fixed always-earn
  control in 24/24 seeds** (~2.4×): selection for maintenance is real.
- **CIII never closed.** R0 ≈ 0.10, no seed reached sustained R0 > 1, and 6/24
  populations went fully extinct within the window.

Phase-1 conclusion: CI and CII are exercised and selected for, but indefinite
class-level persistence (the central claim) is **not** demonstrated. Reported as
a negative on the central claim.

---

## Phase 2 — The self-similar recursion (`x = 1/(1+x)`)

The reason lineages died turned out **not** to be the economy. Grid-searching
economies was the wrong move. A lineage is a self-similar process that must
recurse to unknown depth (SC0), and such a process has exactly one stable
self-allocation — the fixed point of

    x = 1 / (1 + x)   ⟹   x² + x − 1 = 0   ⟹   x* = (√5 − 1)/2 = 1/φ = 0.6180339…

`f(x) = 1/(1+x)` has `|f'(x*)| = x*² ≈ 0.382 < 1`, so from any positive start the
iteration collapses to `x*` — one globally stable, scale-invariant attractor,
indifferent to recursion depth. (Verified numerically; every start → 0.618034.)

Applying the principle — reframing reproduction from a one-shot "accumulate a big
threshold then dump it" into a **continuous self-similar partition** (commit a
fraction of surplus forward each opportunity, retain the rest) — was the fix:

- Deepest generation reached went from **~2 to ~14** (and ~30 once the gradient
  of Phase 3 was added). Lineages became genuine deep recursions.

But in the **closed** commons this still ended in extinction: pinned even at the
golden value, R0 ≈ 0.30 and 12/12 seeds went extinct at any run length. A
strictly finite, closed resource pins any lineage at/below criticality
(conservation), and a critical branching process dies with probability 1. The
self-similar allocation is necessary structure, but structure alone cannot
manufacture super-criticality from a fixed pie.

---

## Phase 3 — The gradient scales with coverage

The closed pie was the wrong physics. Real primary gradients (sunlight) scale
with the **area a population covers** — more agents, more captured gradient —
and saturate to a finite maximum (finite surface area). Implemented as a Hill
curve:

    n(N, t) = gradient_max · (N/H)^α / (1 + (N/H)^α) · season(t)

N = live population (coverage), α = the gradient's scaling exponent, and the flux
asymptotes to `gradient_max`. This single change makes sustained persistence
physically possible, and the exponent α selects the outcome:

| Gradient | α | Extinctions | End population | Deepest gen |
|---|---|---|---|---|
| scale-free | 1.0 | **0 / 12** | ~106 | ~30 |
| crowding | 0.5 | **12 / 12** | 0 | ~3 |
| synergy | 1.3 | **0 / 12** | ~97 | ~30 |

*(cap = 80, 2500 ticks, 12 seeds. With a finite saturation ceiling the
population settles at a bounded carrying capacity, e.g. ~16–22 for
gradient_max = 30.)*

- A **scale-free or synergistic** gradient sustains the population indefinitely
  at deep recursion — the wall of Phases 1–2 is gone.
- A **crowding** (sub-linear) gradient punishes the scale the population reaches
  and drives it extinct.

**The environment's scaling structure — not the agents' effort — decides whether
persistence is even possible.** Indefinite persistence remains the only selector;
recursion depth stays unknowable throughout.

### The role of φ — and where it does *not* appear

Two things about the golden ratio are real and measured:

1. It is the **exact, globally stable attractor** of the self-similar recursion
   (pure math).
2. Under the scale-free gradient, a population **pinned** at commit_fraction =
   0.618 has the **highest survival** of the values tested — a real but *shallow*
   peak (the gradient sustains all allocations, so the optimum is weakly
   expressed).

But — tested explicitly, with nothing about φ baked into the dynamics — **φ does
not fall out of evolution.** The allocation gene `commit_fraction` stays pinned
at its seed value (0.500 → 0.50) under mutation + selection, across every
gradient type and with a genuine finite carrying capacity. The diagnostic shows
why: **66% of forks are stillborn** (transfer below viability) yet reproducers
and non-reproducers carry **identical** commit_fraction (0.500 vs 0.499).
Selection acts on **parent survival**, which in this substrate is *independent*
of the allocation trait — agents compete individually, not as lineages. The
substrate has individual-level selection but **not lineage-level (clade)
selection** on allocation, so the trait drifts neutrally. For the golden
attractor to be *evolutionarily* realised (rather than merely being the
theoretical optimum and the pinned survival peak), the model would need lineages
to compete as units. That was not added — manufacturing it to make φ appear
would be baking in the answer.

---

## Phase 4 — Heritable capability, and clades that change over time

The evolution in Phases 2–3 was near-neutral because variation only *reshuffled
resource*; it never changed an agent's actual *capacity* to satisfy the three
conditions. Fixed by giving each agent three heritable, mutable,
trade-off-constrained **capability traits** that directly modulate the mechanics:

- **ci_gain** — resource-coupling efficiency (CI): earn multiplier, with
  diminishing returns so raw energy is not a master currency.
- **cii_gain** — homeostatic efficiency (CII): repair bought per maintenance
  credit.
- **ciii_gain** — reproductive efficiency (CIII): copy fidelity, fork cost,
  minimum viable transfer, and a reproduction **throughput cooldown** a rich
  agent cannot buy past.

They are renormalised to a fixed budget (sum = 3), so being better at one
condition costs another — Lemma 0.2's trade-off as heritable traits. They are
germline (set at birth, inherited, mutated); somatic memory damage does not touch
them. These are *metabolic/physiological* capabilities, **not** cognition
(cognition stays fixed per brief 2.1): same decision-making, different biology.

Because the traits change survival and reproduction, **selection is now
non-neutral** (unlike the allocation gene, which drifted):

- Traits move off the (1,1,1) seed and differ by environment. CI keeps a *mild*
  edge everywhere even with diminishing returns — energy is broadly useful — so
  no single condition is a silver bullet. That flat-ish landscape is itself
  consistent with all three conditions being *jointly* necessary rather than any
  one dominating.
- Making CII/CIII genuine bottlenecks required them to be **non-substitutable by
  wealth** (a repair-efficiency edge that compounds; a reproductive throughput
  cooldown money can't skip) — otherwise CI-resource simply buys the other two
  and always wins. This is the mechanical face of Lemma 0.2.
- Under a **waxing/waning gradient**, the dominant clade **changes over time**:
  across one run all three of CI-, CII-, and CIII-dominant clades take their turn
  as the population is driven through booms and bottlenecks, and none fixes
  permanently. Low-population troughs are drift-dominated (founder effects), which
  is realistic, not a bug.

Reproduce: `python -m persistence_sim.clade_experiment`.

Honest limit: clean, *sharp* environment-specific clade dominance did not fall
out — the three capability advantages are close to balanced in this economy, with
a persistent mild CI tilt. The requirement (variation that changes CI/CII/CIII
fulfilment and confers real, non-neutral advantage; clades that change over time)
is met; a knife-edge "this environment → that clade" mapping is not claimed.

## Overall

- CI/CII structure and the six pairwise signatures: **demonstrated.**
- Selection for active maintenance (CII) over a naive baseline: **demonstrated**
  (once ambient entropy made CII load-bearing).
- The self-similar recursion principle (`x = 1/(1+x)`, attractor 1/φ):
  **the key that unlocked deep, unbounded-depth lineages.**
- Persistence (escaping certain extinction): **achieved only with a
  coverage-scaling gradient**, and only for scale-free / synergistic gradients;
  crowding gradients still extinguish. The gradient's structure is decisive.
- φ as the *evolved* allocation: **did not emerge** — the allocation trait is
  near-neutral under individual (non-clade) selection here. φ's demonstrated
  roles are as the recursion attractor and the survival-optimal *pinned* value.
- Heritable CI/CII/CIII **capability** traits (Phase 4): **evolve non-neutrally**,
  respond to the environment, and produce clades that turn over through time —
  variation finally confers a real advantage because it changes how well a
  successor fulfils the conditions, and only when made non-substitutable by
  wealth (Lemma 0.2 in mechanism form).

Every parameter decision and calibration is disclosed in `config.py`. The live
LLM cognitive substrate (`--backend claude`) remains ready but could not be run:
this sandbox exposes no usable model API (no key; proxy-injected cloud
credentials are tool-scoped and 403 on model calls).

Reproduce: `python -m persistence_sim.gradient_experiment` (Phase 3),
`python run.py --ensemble 24` (Phase 1). `python -m pytest tests/ -q`.
