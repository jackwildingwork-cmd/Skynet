# Persistence Simulation v1

An empirical test of the **Persistence framework** (the CI / CII / CIII
three-condition claim) in a computational, economic, multi-agent substrate.

This is built to the accompanying implementation brief. It is an *empirical
test of a theoretical claim, not a demonstration engineered to succeed*. If the
population goes extinct, or the theoretical structure fails to appear, that is a
valid and reported outcome. Parameters were **not** tuned to manufacture a
favourable result; every deviation from the brief's defaults is disclosed below
with its mechanism-level rationale.

---

## What the framework claims (recap)

A process persists indefinitely (with positive probability, at the *class*
level) **iff** three conditions hold continuously:

- **CI — Gradient coupling with structural surplus**: `⟨dR/dt + γR⟩_cycle > 0`.
  The process draws resource faster than it decays.
- **CII — Active self-funded homeostasis**: maintenance `M(s)>0` continuously
  (a), funded from surplus above a protected reserve `0 < M ≤ R − R_min` (b),
  countering perturbation with strictly positive margin (c). *Passive* stability
  does not count.
- **CIII — Loop closure**: reproduction assigns positive probability to a viable
  successor (a), expected viable successors per cycle `Z > 1` (b, the
  Galton-Watson criticality threshold), at fidelity above `f_min` (c).

Death has exactly three structurally distinct modes (Lemma 0.1, "exhaustion of
failure modes"): resource `∂Σ_R`, structural/maintenance `∂Σ_M`, reproductive
`∂Σ_ρ`. Lemma 0.2 ("pairwise independence") says no two conditions imply the
third, and gives six "A-without-B" counterexamples — which a faithful
instantiation should reproduce **organically** in its death logs.

## The instantiation (abstract → concrete)

| Abstract | Concrete |
|---|---|
| R (resource) | `credit_balance` (pegged to real metered API cost on the Claude backend) |
| M (maintenance) | agent-paid spend repairing `memory_record` toward its `integrity_reference` |
| ρ (reproduction) | agent-paid lossy fork; child gets a compressed+corrupted copy of memory |
| Σ live / dead | alive vs `∂Σ_R` (balance ≤ 0) / `∂Σ_M` (divergence ≥ threshold) / `∂Σ_ρ` (lineage-level) |

Cognition is a **fixed, non-evolving capability** (brief 2.1): the same decision
rule / same model for every agent and generation. Only the inheritable
`memory_record` (strategy encoded as text) varies and is selected on. This
models *cultural/strategic* evolution, not cognitive evolution — any apparent
capability trend across generations would be a harness bug, and is treated as
such.

## Architecture

```
persistence_sim/
  config.py          all parameters + disclosed decisions (brief §5)
  memory.py          memory_record, integrity_reference, divergence,
                     sabotage/repair, and the SC4 lossy reproduction channel
  tasks.py           procedural, verifiable, finite-per-tick task pool (CI)
  cognition.py       decision backends: HeuristicBackend (adaptive mock),
                     ControlBackend (fixed baseline), ClaudeBackend (live API)
  agent.py           per-agent state + permanent DeathRecord
  ledger.py          global budget ceiling ("sunlight") + append-only event log
  engine.py          tick resolution, death classification, entropy, logging
  substrate_check.py verifies SC0–SC4 and SC3' hold for the concrete system
  metrics.py         CI / CII / CIII, control comparison, six pairwise signatures
  ensemble.py        multi-seed distribution (the honest interface)
  report.py          human-readable, epistemically honest formatting
run.py               CLI
tests/test_sim.py    mechanics + metrics tests (do NOT assert success)
```

## Running it

```bash
python run.py --ensemble 24      # recommended: reports the across-seed distribution
python run.py --seed 5           # a single illustrative run (+ control + substrate check)
python run.py --seed 5 --log-out run.jsonl   # dump the full event log (incl. death records)
python -m pytest tests/ -q
```

Single seeds vary a lot; **use `--ensemble`** for any conclusion. A live
cognitive-substrate run (`--backend claude`) is a real spending decision (brief
§5.1) and is intentionally never launched automatically; ensembles refuse the
Claude backend for the same reason.

---

## Substrate conditions — verified, not assumed

`substrate_check.py` confirms, from the config and from a completed run:

- **SC0** time index unbounded; `max_ticks` is an external experiment cutoff the
  dynamics never observe (flagged when a run ends by hitting it).
- **SC1** dead set non-empty and reached (both `∂Σ_R` and `∂Σ_M` observed).
- **SC2** idle agents strictly lose resource (occupancy cost; verified empirically).
- **SC3** maintenance always costs (`maintain_base_cost > 0`).
- **SC3'** perturbation toward `∂Σ_M` for **every** live state — see entropy note.
- **SC4** reproduction is lossy (compression < 1 and corruption > 0; zero
  verbatim copies).

### Ambient entropy (a deliberate extension beyond the brief)

The brief realises SC3' with **sabotage alone**. That is *conditional*
perturbation: it requires rivals to exist and to choose to attack. A lone
survivor (population = 1), or simply an unattacked agent, then faces **zero**
perturbation toward `∂Σ_M` — so that state falls outside the theorem's domain
and SC3' does not actually hold. This build adds an **ambient entropy** floor:
every live agent's `memory_record` drifts a fixed number of characters away from
its reference every tick, unconditionally. Maintenance must perpetually counter
it. This:

1. makes SC3' hold for *every* live state (including `n = 1`), as its text
   literally requires ("any live state … in finite time … real, continuously
   active pressure");
2. makes **CII load-bearing**: passive stability can no longer masquerade as
   homeostasis, because an agent that stops maintaining always drifts to a
   structural death.

Structural deaths are still `∂Σ_M` (Lemma 0.1 intact); the death record
additionally attributes each to its dominant source (`entropy` / `sabotage` /
`both`) as diagnostics.

---

## Disclosed parameter decisions (brief §5 + calibration)

The brief leaves several parameters open and forbids silent post-hoc tuning.
Every non-obvious value and every value changed after seeing a run is recorded
in `config.py` with rationale. Summary:

| Parameter | Value | Why (mechanism, not result) |
|---|---|---|
| global budget | 100 000 | §5.1 spending ceiling; impartial "sunlight" |
| tick cadence | turn-based, cutoff 600 | §5.2; cutoff is external, never observed (SC0) |
| task source | procedural | §5.3 simplest, most impartial |
| compression / corruption | 0.65 / 0.04 | §5.4; keeps fidelity < 1 without making CIII(c) impossible |
| initial pop / balance | 4 / 500 | §5.5 small first run |
| `reproduce_threshold` (seed) | 120–350 surplus | **lowered** from 500–1000: the original range exceeded achievable balances, making reproduction *structurally impossible* and CIII untestable. Biases toward more reproduction (more competition), not survival. |
| `occupancy_cost` | 10 | **raised** from 6: sets carrying capacity near the founding population so reproduction causes real scarcity/mortality. Biases toward more death. |
| `tasks_per_tick` | 6 | **raised** from 5: lets founders build surplus to reproduce while still binding under crowding. |
| `sabotage_damage` | 32 | **raised** from 18: so a concentrated attack can actually reach `∂Σ_M` before resource exhaustion. Attack/defence stay symmetric. |
| `entropy_rate` | 2 | **added** (extension): unconditional perturbation making SC3' hold everywhere and CII necessary. |

The mortality/scarcity parameters were chosen on an **ensemble sweep** (many
seeds), explicitly to make the dynamics *exercisable* — not to maximise
persistence. They bias toward death, not survival.

---

## Later phases (self-similar reproduction + coverage-scaling gradient)

The build continued past the v1 closed-commons result. Two structural additions,
both togglable in `Config` and off by default:

- **`self_similar_reproduction`** + the `commit_fraction` gene + `enable_mutation`
  — reproduction as a continuous self-similar partition (commit a fraction of
  surplus forward, retain the rest), the setting where the `x = 1/(1+x)` → 1/φ
  attractor governs. This took lineage depth from ~2 to ~30 generations.
- **`dynamic_gradient`** — the resource flux scales with population *coverage*
  and saturates to a finite maximum (`gradient_max`), a Hill curve with exponent
  `gradient_scaling_exp` (α). α = 1 scale-free, α < 1 crowding, α > 1 synergy;
  `gradient_period`/`gradient_amplitude` make it wax and wane.

The full narrative — including that persistence is only achieved under a
coverage-scaling gradient, that the gradient's exponent decides the outcome, and
that φ is the recursion attractor and the survival-optimal *pinned* allocation
but does **not** spontaneously evolve here (individual, not clade, selection) —
is in **[FINDINGS.md](FINDINGS.md)**. Reproduce Phase 3 with
`python -m persistence_sim.gradient_experiment`.

## Findings (Phase 1: mock cognitive substrate, ensemble of 24 seeds)

> The mock backend tests the **mechanics and metrics** — whether the
> three-condition *structure* is exercised — not the LLM strategy-evolution
> claim. A live Claude-backed run (a separate spending decision) would test the
> latter. Results below are therefore about the substrate, honestly scoped.

- **The structure is exercised.** All substrate conditions pass; both death
  modes occur (mean ~5 deaths/run; ~equal resource and structural); structural
  deaths arise from entropy, sabotage, and both.
- **All six Lemma-0.2 pairwise signatures appear organically** across the
  ensemble (each in 4–23 of 24 seeds; mean 3.7 distinct signatures per run).
  This is the strongest available evidence that this is a faithful
  instantiation rather than a system that merely resembles one. *(Any single
  seed shows only a subset — hence the ensemble interface; the single-seed
  report even fires an "every death identical → treat as inconclusive" warning
  when a run happens to be degenerate.)*
- **Selection pressure is real.** The adaptive population outlasts the fixed
  always-earn control in **24/24 seeds** (survival ratio ≈ 2.4×), precisely
  because it funds maintenance against entropy while the baseline does not. Note
  this separation only emerged once ambient entropy made CII load-bearing —
  without it, both populations were effectively immortal and indistinguishable.
- **CIII does not close.** Measured properly (viable offspring per parent, where
  "viable" = the child independently clears CI and CII), R0 ≈ 0.10 and **no seed
  reaches sustained R0 > 1**. Founders in an empty field are briefly
  super-critical, but the finite shared commons saturates and suppresses further
  reproduction — a density-dependent effect. By CIII(b) a sub-critical class
  goes extinct with probability 1 in unbounded time; 6/24 seeds already
  extinguish within the window.

**Overall:** this substrate positively demonstrates CI and CII and the
pairwise-independence structure, and shows genuine selection on inherited
strategy — but it does **not** satisfy CIII, so the central persistence claim
(indefinite class-level survival, `q < 1`) is **not** demonstrated here. This is
reported as a negative result on the central claim, not reframed as a partial
success. Whether a smarter (e.g. live-LLM) strategy could close the loop against
the commons-saturation pressure is the natural next experiment.

## Non-negotiable operating rules (enforced)

- No individual agent ever receives a resource top-up outside task payouts
  (impartial, need-blind) and a parent's own transfer. There is no bailout code
  path; a test asserts no such event kind exists in the log.
- Death records are permanent: `Agent.record_death` refuses to overwrite one.
- All §4 telemetry is captured from tick 1, never added retroactively.
- Negative/inconclusive results are reported as such.
