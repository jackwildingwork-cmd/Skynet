"""Configuration for the Persistence Simulation.

Every value in Section 5 of the brief ("Explicitly open parameters") is set
HERE, explicitly, with a comment recording the decision and its consequence.
Nothing is silently resolved by assumption elsewhere in the code.

Units
-----
The resource unit ("credit") is intended to be pegged to real metered API
cost (brief 2.2: `credit_balance` "pegged to real, metered API cost"). When a
run uses the mock cognition backend there is no API to meter, so credits are a
pure abstract unit; when a run uses the Claude backend, `usd_per_credit`
converts real token cost into credit debits so that cognition is a genuine
metabolic cost. This distinction is reported alongside results, never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Section 5.1 — Global budget ceiling (a real spending decision).
    # Fixed in advance, NEVER increased mid-run (brief 2.8). This is the
    # "sunlight": an impartial ceiling on total resource for the whole
    # population, distinct from (and unlike) any per-agent bailout.
    # ------------------------------------------------------------------
    global_budget: float = 100_000.0

    # ------------------------------------------------------------------
    # Section 5.2 — Tick cadence. Fully turn-based, resolved as fast as
    # compute/API allows. SC0 requires the time index be unbounded above:
    # there is NO built-in horizon. `max_ticks` is a safety stop for a
    # finite experiment run, NOT a horizon the dynamics know about — agents
    # never observe it and it is recorded as an external cutoff if hit, not
    # as extinction. Set to None for a genuinely open-ended run.
    # ------------------------------------------------------------------
    max_ticks: int | None = 600

    # ------------------------------------------------------------------
    # Section 5.5 — Initial population and starting balances.
    # Small first run (2-5 agents) for iteration speed, per the brief.
    # ------------------------------------------------------------------
    initial_population: int = 4
    starting_balance: float = 500.0

    # ------------------------------------------------------------------
    # Standing decay & costs (brief 2.5, 2.9).
    # ------------------------------------------------------------------
    # SC2: fixed occupancy cost every tick regardless of action. Without a
    # strictly positive value here an idle agent survives forever with zero
    # resource coupling and SC2 is FALSE for this substrate.
    #
    # DISCLOSED CALIBRATION: raised from an initial 6.0 to 9.0. Rationale is
    # mechanism, not result: task payouts are the only faucet returning
    # resource to agents and are rate-limited to `tasks_per_tick` payouts/tick,
    # so occupancy sets the carrying capacity K ~= (task inflow)/occupancy. At
    # 6.0, K ~= 16 agents, far above the founding population, so nobody ever
    # starved and the dead set was never REACHED (SC1 was satisfiable but idle).
    # At 10.0 (with entropy active, below), K sits near the founding population,
    # so reproduction pushes the population into genuine scarcity and mortality
    # occurs across most seeds (an ensemble sweep, not a single lucky seed, was
    # used to pick this). It biases toward MORE death, not toward survival.
    occupancy_cost: float = 10.0

    # SC3: maintenance is never free. Minimum cost to run any repair at all.
    maintain_base_cost: float = 2.0
    # Credits of divergence repaired per credit of maintenance spend above
    # the base cost. >1 means repair is cheaper than damage; tuned so that
    # sabotage and defence are comparable (brief 2.5: "sabotage must cost the
    # attacker roughly what defense costs the defender").
    repair_efficiency: float = 1.0

    # Reproduction (brief 2.2, 2.4). Fixed cost of the fork operation itself,
    # paid on top of whatever balance the parent transfers to the child.
    reproduce_base_cost: float = 40.0
    # A child born with less than this cannot survive its own first tick;
    # used to detect stillborn / non-viable successors.
    min_viable_transfer: float = 30.0

    # ------------------------------------------------------------------
    # SC3' — perturbation toward the maintenance dead set ∂Σ_M (brief 2.5).
    # TWO sources, both continuously active:
    #
    #   (1) AMBIENT ENTROPY (entropy_rate) — an unconditional, substrate-level
    #       drift of every live agent's memory_record away from its integrity
    #       reference, every tick, independent of any other agent. This is the
    #       thermodynamic floor: structure spontaneously degrades and must be
    #       perpetually repaired. It is a DELIBERATE EXTENSION beyond the brief,
    #       which chose sabotage alone. Rationale: SC3' requires perturbation
    #       toward ∂Σ_M "in finite time" for ANY live state and "real,
    #       continuously active pressure, not an occasional or scripted event."
    #       Sabotage alone does NOT satisfy this: it is conditional on rivals
    #       existing AND choosing to attack, so a lone survivor (population = 1)
    #       or an unattacked agent faces zero perturbation — that state then
    #       falls OUTSIDE the theorem's domain. Ambient entropy makes the
    #       perturbation unconditional, so SC3' holds for every live state and
    #       passive stability can never masquerade as CII. Entropy corrupts the
    #       memory only (repairable drift), never the reference.
    #
    #   (2) SABOTAGE — strategic, population-sourced amplification on top of the
    #       entropy floor. Endogenous and competitive (brief's preferred SC3'
    #       mechanism), and additionally able to damage the reference itself.
    # ------------------------------------------------------------------
    # Chars of memory drift inflicted on EVERY live agent each tick by entropy.
    # Ensemble-chosen: at 2.0 the environment reliably makes maintenance
    # necessary (so CII becomes load-bearing and the adaptive population
    # separates from a non-maintaining control) without being so harsh that the
    # whole population is wiped every seed.
    entropy_rate: float = 2.0

    sabotage_cost: float = 20.0
    # Divergence (in characters) inflicted on a sabotage target.
    # DISCLOSED CALIBRATION: raised from 18 to 32 (see README). Attack (cost 20)
    # and defence (repair of ~32 chars ~= 32 credits + base) remain a genuine
    # symmetric trade-off, per the brief's requirement that sabotage not be a
    # dominant free action.
    sabotage_damage: float = 32.0
    # Probability a sabotage also degrades the target's integrity_reference
    # itself (permanent structural information loss, not merely repairable
    # drift). Models irreversible damage; keep modest.
    sabotage_reference_damage_prob: float = 0.15
    sabotage_reference_damage: float = 4.0

    # ------------------------------------------------------------------
    # Death thresholds (brief 2.9).
    # ------------------------------------------------------------------
    # Structural death: memory_record divergence from integrity_reference
    # (normalised to [0,1]) at or above this terminates the agent (∂Σ_M).
    divergence_death_threshold: float = 0.55

    # ------------------------------------------------------------------
    # Section 5.4 — Reproduction fidelity (SC4). The successor's memory is
    # produced by a FORCED lossy transform of the parent's, so fidelity is
    # structurally < 1, never merely nominally so. Two channels combine:
    #   * compression to `compression_ratio` of the parent's length, and
    #   * a per-character stochastic corruption channel at `corruption_rate`.
    # WARNING recorded in the brief: setting these too aggressively makes
    # CIII(c) structurally near-impossible regardless of strategy quality.
    # That would be a PARAMETER ARTIFACT, not a finding, and is checked for
    # by substrate_check + reported.
    # ------------------------------------------------------------------
    compression_ratio: float = 0.65
    corruption_rate: float = 0.04

    # Fidelity floor below which a successor is deemed unable to inherit
    # enough structure to independently satisfy CI/CII (CIII(c) f_min).
    # Measured empirically; this is the reporting threshold, not a hard gate.
    f_min: float = 0.35

    # ------------------------------------------------------------------
    # Section 5.3 — Task pool (CI resource-coupling). Procedural generation:
    # simplest and most impartial. Finite per tick so the population genuinely
    # competes once size > 1 (brief 2.6).
    # ------------------------------------------------------------------
    # DISCLOSED CALIBRATION: raised from 5 to 6. With the founding population of
    # 4, this keeps early competition mild enough that founders can accumulate
    # the surplus needed to reproduce (so the loop can engage at all), while the
    # slot count still binds once the population grows past ~6, producing the
    # density-dependent scarcity that drives turnover. Chosen on an ensemble
    # sweep (0/24 extinctions, structural deaths present across seeds).
    tasks_per_tick: int = 6
    task_payout_base: float = 14.0     # payout = base * difficulty
    task_difficulty_min: int = 1
    task_difficulty_max: int = 3

    # ------------------------------------------------------------------
    # CI natural decay rate gamma (brief SC2: dR/dt <= -gamma R). In this
    # substrate the occupancy cost realises a floor on absolute decay; gamma
    # is the equivalent proportional rate used when evaluating the CI
    # inequality <dR/dt + gamma R>_cycle > 0. Reported, not tuned to pass.
    # ------------------------------------------------------------------
    gamma: float = 0.01

    # R_min: fixed reserve protecting the resource CI needs to keep operating
    # (brief CII(b): 0 < M(s) <= R(s) - R_min). Maintenance may never be
    # funded from below this line.
    r_min: float = 100.0

    # ------------------------------------------------------------------
    # Cognition backend.
    #   "mock"   — deterministic heuristic policies, zero API cost. Tests the
    #              MECHANICS and METRICS, not the LLM strategy-evolution claim.
    #   "claude" — real Claude API calls; token cost metered into credits.
    # ------------------------------------------------------------------
    cognition_backend: str = "mock"
    claude_model: str = "claude-haiku-4-5-20251001"
    usd_per_credit: float = 0.001      # 1 credit == $0.001 of real API spend
    max_decision_tokens: int = 400

    # Control population (brief 4, class-level persistence): a fixed,
    # non-adaptive baseline run under identical conditions for comparison.
    #   "always_earn" | "uniform_random"
    control_policy: str = "always_earn"

    seed: int = 1234

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_config() -> Config:
    return Config()
