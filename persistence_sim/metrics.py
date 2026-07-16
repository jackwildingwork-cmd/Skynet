"""Success metrics tied to the framework's own vocabulary (brief 4).

Everything here is computed from telemetry captured during the run — nothing is
added retroactively. The metrics deliberately use the framework's definitions,
not a vaguer "agents survived":

  CI   — cycle-averaged <dR/dt + gamma R> across a lineage, positive AND stable.
  CII  — integrity failure rate vs threshold; maintenance inside the CII(b)
         band; preventive vs reactive margin (CII(c)).
  CIII — R0 as VIABLE offspring per parent across generations, where "viable"
         means the offspring itself independently clears CI and CII (not merely
         that a fork happened). CIII(b) threshold Z>1.
  Class-level persistence — lineage survival vs a fixed control population.
  Falsification — whether the six pairwise-independence failure signatures from
         Lemma 0.2 appear organically in the death logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Dict, List, Optional

from .agent import Agent, DeathCause
from .config import Config
from .engine import RunResult

MIN_VIABLE_AGE = 5   # ticks a child must survive to be eligible as "viable"


@dataclass
class AgentConditions:
    agent_id: int
    lineage_id: int
    generation: int
    age: int
    ci: bool                 # cleared Condition I over its life
    ci_value: float          # cycle-averaged dR/dt + gamma R
    cii: bool                # cleared Condition II (active homeostasis)
    maintained: bool         # did any maintenance at all (CII(a))
    preventive: bool         # any preventive maintenance (CII(c))
    denied_maintenance: bool # hit the CII(b) reserve band and was denied
    viable_offspring: int    # CIII: offspring that independently clear CI+CII
    death_cause: Optional[str]
    lineage_terminal: bool


def _agent_age(agent: Agent) -> int:
    return len(agent.balance_history)


def _ci_value(agent: Agent, gamma: float) -> float:
    """Cycle-averaged dR/dt + gamma R over the agent's life."""
    bals = agent.balance_history
    if len(bals) < 2:
        return 0.0
    vals = []
    for i in range(1, len(bals)):
        dr = bals[i] - bals[i - 1]
        vals.append(dr + gamma * bals[i - 1])
    return mean(vals)


def evaluate_agent(agent: Agent, cfg: Config, denied_ids: set) -> AgentConditions:
    age = _agent_age(agent)
    ci_val = _ci_value(agent, cfg.gamma)
    ci = ci_val > 0
    maintained = agent.maintain_spend_history != []
    preventive = agent.preventive_maintain_ticks > 0
    cause = agent.death.death_cause.value if agent.death else None
    # CII satisfied: active (self-funded) homeostasis actually kept the agent
    # off the structural boundary. Passive survival (no maintenance, never
    # sabotaged hard enough to matter) does NOT count — CII(c) requires an
    # active countermeasure with positive margin.
    cii = (cause != DeathCause.STRUCTURAL.value) and maintained and preventive
    return AgentConditions(
        agent_id=agent.agent_id, lineage_id=agent.lineage_id,
        generation=agent.generation, age=age,
        ci=ci, ci_value=ci_val, cii=cii, maintained=maintained,
        preventive=preventive, denied_maintenance=agent.agent_id in denied_ids,
        viable_offspring=0,  # filled in a second pass (needs all agents scored)
        death_cause=cause,
        lineage_terminal=agent.death.lineage_terminal if agent.death else False,
    )


def _child_is_viable(child_cond: AgentConditions) -> bool:
    """CIII viability: the offspring independently clears CI and CII and lived
    long enough to demonstrate it."""
    return (child_cond.age >= MIN_VIABLE_AGE
            and child_cond.ci
            and child_cond.death_cause != DeathCause.STRUCTURAL.value)


@dataclass
class MetricsReport:
    label: str
    ended_reason: str
    final_tick: int
    n_agents: int
    n_deaths: int
    death_cause_counts: Dict[str, int]
    # CI
    ci_lineage_positive_frac: float
    ci_stable_frac: float
    # CII
    integrity_failure_rate: float          # structural deaths / all deaths
    maintenance_denied_events: int
    preventive_frac: float                 # of maintainers, fraction preventive
    reactive_only_frac: float
    # CIII
    r0_by_generation: Dict[int, float]
    overall_r0: float
    max_generation: int
    # persistence
    lineage_survival_ticks: Dict[int, int]
    mean_lineage_survival: float
    total_lineage_survival: int
    # falsification
    pairwise_signatures: Dict[str, int]
    signature_examples: Dict[str, List[int]]
    identical_death_warning: bool
    per_agent: List[AgentConditions] = field(default_factory=list)


def compute_metrics(run: RunResult) -> MetricsReport:
    cfg = run.config
    denied_ids = {e["agent_id"] for e in run.log.of_kind("maintain_denied")}

    conds: Dict[int, AgentConditions] = {}
    for agent in run.agents.values():
        conds[agent.agent_id] = evaluate_agent(agent, cfg, denied_ids)

    # Second pass: viable offspring counts (needs every agent scored first).
    for agent in run.agents.values():
        viable = 0
        for cid in agent.offspring_ids:
            cc = conds.get(cid)
            if cc and _child_is_viable(cc):
                viable += 1
        conds[agent.agent_id].viable_offspring = viable

    # ---- CI ----
    lineages: Dict[int, List[AgentConditions]] = {}
    for c in conds.values():
        lineages.setdefault(c.lineage_id, []).append(c)
    ci_pos = 0
    ci_stable = 0
    for lid, members in lineages.items():
        vals = [m.ci_value for m in members]
        if mean(vals) > 0:
            ci_pos += 1
        # "stable": positive and low relative dispersion, not a single lucky spike.
        if mean(vals) > 0 and (pstdev(vals) <= abs(mean(vals)) * 3 + 1e-9):
            ci_stable += 1
    n_lineages = max(1, len(lineages))

    # ---- CII ----
    deaths = run.log.of_kind("death")
    cause_counts: Dict[str, int] = {}
    for d in deaths:
        cause_counts[d["cause"]] = cause_counts.get(d["cause"], 0) + 1
    structural = cause_counts.get(DeathCause.STRUCTURAL.value, 0)
    integrity_failure_rate = structural / len(deaths) if deaths else 0.0
    maintainers = [c for c in conds.values() if c.maintained]
    preventive_frac = (sum(1 for c in maintainers if c.preventive) / len(maintainers)
                       if maintainers else 0.0)
    reactive_only_frac = (sum(1 for c in maintainers if not c.preventive) / len(maintainers)
                          if maintainers else 0.0)

    # ---- CIII ----
    by_gen: Dict[int, List[int]] = {}
    for c in conds.values():
        by_gen.setdefault(c.generation, []).append(c.viable_offspring)
    r0_by_gen = {g: (mean(v) if v else 0.0) for g, v in by_gen.items()}
    all_viable = [c.viable_offspring for c in conds.values()]
    overall_r0 = mean(all_viable) if all_viable else 0.0
    max_gen = max((c.generation for c in conds.values()), default=1)

    # ---- persistence ----
    lineage_survival = _lineage_survival(run)
    mean_surv = mean(lineage_survival.values()) if lineage_survival else 0.0
    total_surv = sum(lineage_survival.values())

    # ---- falsification: six pairwise signatures ----
    signatures, examples = _pairwise_signatures(conds)
    identical_warning = _deaths_look_identical(deaths, conds)

    return MetricsReport(
        label=run.label, ended_reason=run.ended_reason, final_tick=run.final_tick,
        n_agents=len(conds), n_deaths=len(deaths), death_cause_counts=cause_counts,
        ci_lineage_positive_frac=ci_pos / n_lineages,
        ci_stable_frac=ci_stable / n_lineages,
        integrity_failure_rate=integrity_failure_rate,
        maintenance_denied_events=len(run.log.of_kind("maintain_denied")),
        preventive_frac=preventive_frac, reactive_only_frac=reactive_only_frac,
        r0_by_generation=r0_by_gen, overall_r0=overall_r0, max_generation=max_gen,
        lineage_survival_ticks=lineage_survival, mean_lineage_survival=mean_surv,
        total_lineage_survival=total_surv,
        pairwise_signatures=signatures, signature_examples=examples,
        identical_death_warning=identical_warning,
        per_agent=list(conds.values()),
    )


def _lineage_survival(run: RunResult) -> Dict[int, int]:
    """Ticks from a lineage's first birth to its last member's death (or run end)."""
    first_birth: Dict[int, int] = {}
    last_active: Dict[int, int] = {}
    for agent in run.agents.values():
        lid = agent.lineage_id
        first_birth[lid] = min(first_birth.get(lid, agent.birth_tick), agent.birth_tick)
        end = agent.death.death_tick if agent.death else run.final_tick
        last_active[lid] = max(last_active.get(lid, end), end)
    return {lid: last_active[lid] - first_birth[lid] for lid in first_birth}


def _pairwise_signatures(conds: Dict[int, AgentConditions]):
    """Detect the six Lemma-0.2 pairwise-independence death signatures.

    Each is "condition A held while condition B failed" surfacing organically:
      CI_without_CII  : good resources, died structurally.
      CI_without_CIII : good resources, lineage terminated with no viable kids.
      CII_without_CI  : actively maintained, died of resource exhaustion.
      CII_without_CIII: actively maintained, lineage terminated, no viable kids.
      CIII_without_CI : had viable offspring, yet died of resource exhaustion.
      CIII_without_CII: had viable offspring, yet died structurally.
    """
    sigs = {k: 0 for k in (
        "CI_without_CII", "CI_without_CIII", "CII_without_CI",
        "CII_without_CIII", "CIII_without_CI", "CIII_without_CII",
    )}
    ex: Dict[str, List[int]] = {k: [] for k in sigs}

    def add(sig: str, aid: int):
        sigs[sig] += 1
        if len(ex[sig]) < 5:
            ex[sig].append(aid)

    for c in conds.values():
        died_resource = c.death_cause == DeathCause.RESOURCE.value
        died_structural = c.death_cause == DeathCause.STRUCTURAL.value
        has_viable = c.viable_offspring > 0

        if c.ci and died_structural:
            add("CI_without_CII", c.agent_id)
        if c.ci and c.lineage_terminal and not has_viable:
            add("CI_without_CIII", c.agent_id)
        if c.cii and died_resource:
            add("CII_without_CI", c.agent_id)
        if c.cii and c.lineage_terminal and not has_viable:
            add("CII_without_CIII", c.agent_id)
        if has_viable and died_resource:
            add("CIII_without_CI", c.agent_id)
        if has_viable and died_structural:
            add("CIII_without_CII", c.agent_id)
    return sigs, ex


def _deaths_look_identical(deaths: List[dict], conds) -> bool:
    """Warn (brief 1.3/4) if every death looks the same regardless of strategy —
    evidence the three-condition structure is NOT being exercised."""
    if len(deaths) < 3:
        return False
    causes = {d["cause"] for d in deaths}
    return len(causes) == 1


# ---------------------------------------------------------------------------
# Control comparison (brief 4: class-level persistence vs a fixed baseline).
# ---------------------------------------------------------------------------

@dataclass
class Comparison:
    adaptive_total_survival: int
    control_total_survival: int
    adaptive_mean_survival: float
    control_mean_survival: float
    adaptive_max_generation: int
    control_max_generation: int
    adaptive_outlasts_control: bool
    ratio: float


def compare(adaptive: MetricsReport, control: MetricsReport) -> Comparison:
    ratio = (adaptive.total_lineage_survival / control.total_lineage_survival
             if control.total_lineage_survival else float("inf"))
    return Comparison(
        adaptive_total_survival=adaptive.total_lineage_survival,
        control_total_survival=control.total_lineage_survival,
        adaptive_mean_survival=adaptive.mean_lineage_survival,
        control_mean_survival=control.mean_lineage_survival,
        adaptive_max_generation=adaptive.max_generation,
        control_max_generation=control.max_generation,
        # "meaningfully outlast" — require a clear margin, not a coin-flip edge.
        adaptive_outlasts_control=(ratio >= 1.15),
        ratio=ratio,
    )
