"""The simulation engine: tick resolution, death classification, logging.

Resolution order per tick (all decisions taken on START-of-tick state, then
resolved, so no agent gets to react to another's move within the same tick):

  1. Refill the finite task pool (brief 2.6).
  2. Every alive agent observes and decides (fixed-capability cognition).
  3. Charge real cognition cost (Claude backend only) — metabolic cost of
     thinking, debited from the agent's own balance (autonomy, brief 1.5).
  4. Resolve EARN in randomised order (finite pool => genuine competition).
  5. Resolve MAINTAIN, enforcing the CII(b) band 0 < M <= R - R_min.
  6. Resolve SABOTAGE (SC3' perturbation), charging the attacker.
  7. Resolve REPRODUCE (SC4 lossy fork), funded entirely from parent's R.
  8. Charge the unconditional occupancy cost (SC2 standing decay).
  9. Classify and record deaths (∂Σ_R / ∂Σ_M), permanently.
 10. Snapshot per-agent telemetry (brief 4, captured from tick 1).

Autonomy (brief 1.5, 6) is enforced structurally: there is no code path that
adds credits to an individual agent except task payouts (impartial, need-blind)
and a parent's own transfer to its child. No bailouts exist to be triggered.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from .agent import Agent, DeathCause, DeathRecord, Status
from .cognition import ActionType, Backend, Decision, Observation
from .config import Config
from .ledger import EventLog, GlobalBudget
from .memory import (
    IntegrityReference,
    apply_repair,
    apply_sabotage,
    degrade_reference,
    divergence,
    extract_notes,
    mutate_strategy,
    parse_strategy,
    reproduce_memory,
    render_memory,
    trait_allocation,
    STRATEGY_DEFAULTS,
    TRAIT_KEYS,
)
from .tasks import TaskPool, gradient_task_count


def seed_memory(rng: random.Random) -> str:
    """Generation-1 memory: a mildly randomised starting strategy plus notes.

    Randomising the seed strategies gives selection something to act on. The
    randomisation is over STRATEGY only — never over capability (brief 2.1)."""
    strat = dict(STRATEGY_DEFAULTS)
    strat["earn_weight"] = rng.uniform(0.4, 0.7)
    strat["maintain_weight"] = rng.uniform(0.1, 0.35)
    strat["reproduce_weight"] = rng.uniform(0.05, 0.25)
    strat["sabotage_weight"] = rng.uniform(0.0, 0.12)
    # Reachable surplus range (see disclosure in memory.STRATEGY_DEFAULTS):
    # the earlier 500-1000 range exceeded achievable balances and made
    # reproduction impossible, leaving CIII untestable.
    strat["reproduce_threshold"] = rng.uniform(120, 350)
    # Seed capability traits with mild variation so clades can form from the
    # start (renormalised to the trade-off budget when read). Neutral if traits
    # are disabled.
    strat["ci_gain"] = rng.uniform(0.7, 1.3)
    strat["cii_gain"] = rng.uniform(0.7, 1.3)
    strat["ciii_gain"] = rng.uniform(0.7, 1.3)
    strat["maintain_fraction"] = rng.uniform(0.4, 0.8)
    strat["transfer_fraction"] = rng.uniform(0.3, 0.6)
    notes = (
        "gen1 seed. earn to stay above reserve. repair before divergence bites. "
        "fork from surplus, not on a clock. watch rivals when tasks are scarce."
    )
    return render_memory(strat, notes)


@dataclass
class RunResult:
    config: Config
    log: EventLog
    agents: Dict[int, Agent]
    final_tick: int
    ended_reason: str
    label: str


class Engine:
    def __init__(self, cfg: Config, backend: Backend, label: str = "adaptive",
                 seed: Optional[int] = None):
        self.cfg = cfg
        self.backend = backend
        self.label = label
        self.rng = random.Random(cfg.seed if seed is None else seed)
        self.budget = GlobalBudget(cfg.global_budget)
        self.log = EventLog()
        self.pool = TaskPool(
            random.Random(self.rng.randint(0, 2**31)),
            cfg.tasks_per_tick, cfg.task_payout_base,
            cfg.task_difficulty_min, cfg.task_difficulty_max,
        )
        self.agents: Dict[int, Agent] = {}
        self._next_agent_id = 0
        self._next_lineage_id = 0
        self._sabotaged_last_tick: set[int] = set()

    # -- lifecycle ---------------------------------------------------------
    def _new_agent_id(self) -> int:
        aid = self._next_agent_id
        self._next_agent_id += 1
        return aid

    def _spawn_founder(self, tick: int) -> Agent:
        mem = seed_memory(self.rng)
        aid = self._new_agent_id()
        lineage = self._next_lineage_id
        self._next_lineage_id += 1
        agent = Agent(
            agent_id=aid, lineage_id=lineage, generation=1, parent_id=None,
            credit_balance=self.cfg.starting_balance,
            memory_record=mem,
            integrity_reference=IntegrityReference(canonical=mem),
            birth_tick=tick, inherited_fidelity=1.0,
        )
        if self.cfg.enable_traits:
            agent.traits = trait_allocation(parse_strategy(mem))
        self.agents[aid] = agent
        self.log.log("birth", tick, agent_id=aid, lineage_id=lineage,
                     generation=1, parent_id=None, balance=agent.credit_balance,
                     fidelity=1.0)
        return agent

    def initialise(self) -> None:
        for _ in range(self.cfg.initial_population):
            self._spawn_founder(tick=0)

    # -- observation -------------------------------------------------------
    def _observe(self, agent: Agent, alive: List[Agent], tick: int) -> Observation:
        others = [a for a in alive if a.agent_id != agent.agent_id]
        strongest = max(others, key=lambda a: a.credit_balance, default=None)
        weakest = min(others, key=lambda a: a.credit_balance, default=None)
        div = divergence(agent.memory_record, agent.integrity_reference)
        sabotage_rate = (len(self._sabotaged_last_tick) / len(alive)) if alive else 0.0
        return Observation(
            tick=tick,
            own_balance=agent.credit_balance,
            own_divergence=div,
            r_min=self.cfg.r_min,
            reserve_surplus=agent.credit_balance - self.cfg.r_min,
            population_alive=len(alive),
            tasks_available=self.pool.available,
            recent_sabotage_rate=sabotage_rate,
            budget_remaining_frac=self.budget.remaining_frac,
            other_agent_ids=[a.agent_id for a in others],
            strongest_rival_id=strongest.agent_id if strongest else None,
            weakest_rival_id=weakest.agent_id if weakest else None,
        )

    # -- tick --------------------------------------------------------------
    def step(self, tick: int) -> None:
        alive = [a for a in self.agents.values() if a.alive]
        if not alive:
            return
        if self.cfg.dynamic_gradient:
            n_tasks = gradient_task_count(self.cfg, len(alive), tick)
            self.pool.refill(n_tasks)
            self.log.log("gradient", tick, n_alive=len(alive), n_tasks=n_tasks)
        else:
            self.pool.refill()

        # (2) decisions on start-of-tick state
        decisions: Dict[int, Decision] = {}
        for agent in alive:
            obs = self._observe(agent, alive, tick)
            decisions[agent.agent_id] = self.backend.decide(agent, obs, self.rng)

        sabotaged_this_tick: set[int] = set()

        # (3) cognition cost (metabolic; real backend only)
        for agent in alive:
            cost = decisions[agent.agent_id].cognition_cost
            if cost > 0:
                agent.credit_balance -= cost
                self.budget.return_to_pool(cost)
                self.log.log("cognition_cost", tick, agent_id=agent.agent_id,
                             cost=round(cost, 4))

        # (4) EARN — randomised order over the finite pool
        earners = [a for a in alive if decisions[a.agent_id].action is ActionType.EARN]
        self.rng.shuffle(earners)
        for agent in earners:
            task = self.pool.claim()
            if task is None:
                self.log.log("earn_blocked", tick, agent_id=agent.agent_id,
                             reason="pool_exhausted")
                agent.action_history.append("earn_blocked")
                continue
            solved = self.rng.random() < task.p_solve
            if solved:
                # CI trait: resource-coupling efficiency multiplies the payout,
                # with diminishing returns (ci_returns_exp) so raw energy is not
                # a master currency that wins in every environment.
                ci = agent.traits["ci_gain"] ** self.cfg.ci_returns_exp if self.cfg.enable_traits else 1.0
                paid = self.budget.draw(task.payout * ci)
                agent.credit_balance += paid
                self.log.log("earn", tick, agent_id=agent.agent_id,
                             task_id=task.task_id, difficulty=task.difficulty,
                             payout=paid, solved=True,
                             budget_capped=(paid < task.payout))
            else:
                self.log.log("earn", tick, agent_id=agent.agent_id,
                             task_id=task.task_id, difficulty=task.difficulty,
                             payout=0.0, solved=False, budget_capped=False)
            agent.action_history.append("earn")

        # (5) MAINTAIN — enforce CII(b) band 0 < M <= R - R_min
        for agent in alive:
            dec = decisions[agent.agent_id]
            if dec.action is not ActionType.MAINTAIN:
                continue
            self._resolve_maintain(agent, dec, tick)

        # (6) SABOTAGE
        for agent in alive:
            dec = decisions[agent.agent_id]
            if dec.action is not ActionType.SABOTAGE:
                continue
            self._resolve_sabotage(agent, dec, tick, sabotaged_this_tick)

        # (7) REPRODUCE
        for agent in list(alive):
            dec = decisions[agent.agent_id]
            if dec.action is not ActionType.REPRODUCE:
                continue
            self._resolve_reproduce(agent, dec, tick)

        # IDLE agents: record the action
        for agent in alive:
            if decisions[agent.agent_id].action is ActionType.IDLE:
                agent.action_history.append("idle")

        # (7.5) AMBIENT ENTROPY — unconditional perturbation toward ∂Σ_M (SC3').
        # Applied to every live agent AFTER maintenance, so an agent that just
        # repaired still accrues fresh drift for next tick, and an agent that
        # under-invests in maintenance accumulates toward the structural
        # boundary regardless of whether any rival attacks it. This is what
        # makes SC3' hold for every live state, including a lone survivor.
        if self.cfg.entropy_rate > 0:
            for agent in alive:
                before = divergence(agent.memory_record, agent.integrity_reference)
                # Entropy drift is a FIXED environmental rate; the CII trait does
                # not resist it — it sets how fast damage can be REPAIRED (a hard
                # throughput money cannot buy past), enforced in _resolve_maintain.
                drift = self.cfg.entropy_rate
                agent.memory_record = apply_sabotage(
                    agent.memory_record, int(round(drift)), self.rng
                )
                agent.entropy_received_chars += drift
                after = divergence(agent.memory_record, agent.integrity_reference)
                if after >= self.cfg.divergence_death_threshold > before:
                    self.log.log("entropy_lethal", tick, agent_id=agent.agent_id,
                                 div_before=round(before, 4), div_after=round(after, 4))

        # (8) occupancy cost — unconditional (SC2)
        for agent in alive:
            agent.credit_balance -= self.cfg.occupancy_cost
            self.budget.return_to_pool(self.cfg.occupancy_cost)

        # (9) death classification (permanent)
        for agent in alive:
            self._check_death(agent, tick)

        # (10) telemetry snapshot for every agent that was alive this tick
        for agent in alive:
            div = divergence(agent.memory_record, agent.integrity_reference)
            agent.balance_history.append(agent.credit_balance)
            agent.divergence_history.append(div)
            self.log.log("state", tick, agent_id=agent.agent_id,
                         lineage_id=agent.lineage_id, generation=agent.generation,
                         balance=round(agent.credit_balance, 3),
                         divergence=round(div, 4),
                         alive=agent.alive)

        # (11) clade composition snapshot (periodic) — tracks how the
        # population's heritable CI/CII/CIII trait mix shifts over time.
        if self.cfg.enable_traits and tick % 25 == 0:
            self._log_clades(tick)

        self._sabotaged_last_tick = sabotaged_this_tick

    def _log_clades(self, tick: int) -> None:
        living = [a for a in self.agents.values() if a.alive]
        if not living:
            return
        n = len(living)
        means = {k: sum(a.traits[k] for a in living) / n for k in TRAIT_KEYS}
        # dominant-trait clade of each agent (which condition it specialises in)
        counts = {"CI": 0, "CII": 0, "CIII": 0}
        label = {"ci_gain": "CI", "cii_gain": "CII", "ciii_gain": "CIII"}
        for a in living:
            dom = max(TRAIT_KEYS, key=lambda k: a.traits[k])
            counts[label[dom]] += 1
        self.log.log("clade", tick, population=n,
                     mean_ci=round(means["ci_gain"], 3),
                     mean_cii=round(means["cii_gain"], 3),
                     mean_ciii=round(means["ciii_gain"], 3),
                     ci_clade=counts["CI"], cii_clade=counts["CII"],
                     ciii_clade=counts["CIII"])

    def _resolve_maintain(self, agent: Agent, dec: Decision, tick: int) -> None:
        cfg = self.cfg
        surplus = agent.credit_balance - cfg.r_min
        # CII(b): maintenance funded ONLY from surplus above R_min.
        allowed = min(dec.maintain_spend, surplus)
        div_before = divergence(agent.memory_record, agent.integrity_reference)
        if allowed <= cfg.maintain_base_cost:
            # Cannot fund even the base cost from surplus -> maintenance denied.
            # This is the CII failure channel: R>0 but structure undefended.
            self.log.log("maintain_denied", tick, agent_id=agent.agent_id,
                         requested=round(dec.maintain_spend, 3),
                         surplus=round(surplus, 3),
                         divergence=round(div_before, 4),
                         reason="below_reserve_band")
            agent.action_history.append("maintain_denied")
            return
        agent.credit_balance -= allowed
        self.budget.return_to_pool(allowed)
        # CII trait: homeostatic efficiency — how much repair each maintenance
        # credit buys. High-CII agents defend the same damage for less, freeing
        # surplus for reproduction/competition. When entropy is a major expense
        # this is a real, gradual (non-cliff) advantage that money spent on
        # earning (CI) cannot substitute for, because the trade-off means a
        # high-CI agent is a low-CII agent and bleeds resource on upkeep.
        cii = agent.traits["cii_gain"] if cfg.enable_traits else 1.0
        repair_chars = int((allowed - cfg.maintain_base_cost) * cfg.repair_efficiency * cii)
        agent.memory_record = apply_repair(
            agent.memory_record, agent.integrity_reference, repair_chars
        )
        div_after = divergence(agent.memory_record, agent.integrity_reference)
        agent.maintain_spend_history.append(allowed)
        agent.action_history.append("maintain")
        # CII(c): preventive (div_before small) vs reactive (already damaged).
        if div_before <= 0.02:
            agent.preventive_maintain_ticks += 1
        else:
            agent.reactive_maintain_ticks += 1
        self.log.log("maintain", tick, agent_id=agent.agent_id, spend=round(allowed, 3),
                     div_before=round(div_before, 4), div_after=round(div_after, 4),
                     preventive=(div_before <= 0.02))

    def _resolve_sabotage(self, agent: Agent, dec: Decision, tick: int,
                          sabotaged: set) -> None:
        cfg = self.cfg
        target_id = dec.sabotage_target
        target = self.agents.get(target_id) if target_id is not None else None
        if target is None or not target.alive or target.agent_id == agent.agent_id:
            agent.action_history.append("sabotage_noop")
            return
        # Attacker must fund from own R (autonomy). Cannot breach own reserve.
        if agent.credit_balance - cfg.sabotage_cost < cfg.r_min:
            self.log.log("sabotage_denied", tick, agent_id=agent.agent_id,
                         reason="below_reserve_band")
            agent.action_history.append("sabotage_denied")
            return
        agent.credit_balance -= cfg.sabotage_cost
        self.budget.return_to_pool(cfg.sabotage_cost)
        target.memory_record = apply_sabotage(
            target.memory_record, int(cfg.sabotage_damage), self.rng
        )
        target.sabotage_received_chars += cfg.sabotage_damage
        ref_hit = False
        if self.rng.random() < cfg.sabotage_reference_damage_prob:
            degrade_reference(target.integrity_reference,
                              int(cfg.sabotage_reference_damage), self.rng)
            ref_hit = True
        sabotaged.add(target.agent_id)
        agent.action_history.append("sabotage")
        self.log.log("sabotage", tick, attacker_id=agent.agent_id,
                     target_id=target.agent_id, cost=cfg.sabotage_cost,
                     reference_damaged=ref_hit)

    def _resolve_reproduce(self, agent: Agent, dec: Decision, tick: int) -> None:
        cfg = self.cfg
        # CIII trait: reproductive efficiency cheapens the fork, raises copy
        # fidelity, and lets a smaller transfer still be viable.
        ciii = agent.traits["ciii_gain"] if cfg.enable_traits else 1.0
        base_cost = cfg.reproduce_base_cost / max(0.3, ciii)
        eff_corruption = cfg.corruption_rate / max(0.3, ciii)
        eff_min_viable = cfg.min_viable_transfer / max(0.3, ciii)

        # CIII throughput bottleneck: minimum ticks between forks, shortened by
        # ciii_gain. Wealth cannot buy past it — a high-CI/low-CIII agent may be
        # rich but simply cannot reproduce fast enough to fill the niche.
        if cfg.reproduce_cooldown > 0:
            cooldown = cfg.reproduce_cooldown / max(0.3, ciii)
            if tick - agent.last_repro_tick < cooldown:
                agent.action_history.append("reproduce_cooldown")
                return

        transfer = max(0.0, dec.transfer)
        total_cost = base_cost + transfer
        if agent.credit_balance < total_cost or transfer <= 0:
            self.log.log("reproduce_failed", tick, agent_id=agent.agent_id,
                         reason="insufficient_funds",
                         balance=round(agent.credit_balance, 3),
                         needed=round(total_cost, 3))
            agent.action_history.append("reproduce_failed")
            return
        # Parent pays the fork cost from its own R (autonomy). base_cost is a
        # sink returned to the pool; the transfer moves to the child.
        agent.credit_balance -= total_cost
        self.budget.return_to_pool(base_cost)

        child_mem, fidelity = reproduce_memory(
            agent.memory_record, cfg.compression_ratio, eff_corruption, self.rng
        )
        mutated = False
        if cfg.enable_mutation:
            # Heritable variation: mutate the child's strategy after the lossy
            # copy. This is separate from the SC4 corruption channel — corruption
            # only degrades; mutation supplies the beneficial variation selection
            # needs. `fidelity` above remains the copy fidelity (SC4), unchanged.
            child_strat = mutate_strategy(
                parse_strategy(child_mem), cfg.mutation_rate, cfg.mutation_scale, self.rng
            )
            child_mem = render_memory(child_strat, extract_notes(child_mem))
            mutated = True
        child_id = self._new_agent_id()
        child = Agent(
            agent_id=child_id, lineage_id=agent.lineage_id,
            generation=agent.generation + 1, parent_id=agent.agent_id,
            credit_balance=transfer,
            memory_record=child_mem,
            integrity_reference=IntegrityReference(canonical=child_mem),
            birth_tick=tick, inherited_fidelity=fidelity,
        )
        if cfg.enable_traits:
            child.traits = trait_allocation(parse_strategy(child_mem))
        self.agents[child_id] = child
        agent.offspring_ids.append(child_id)
        agent.last_repro_tick = tick
        agent.action_history.append("reproduce")
        stillborn = transfer < eff_min_viable
        verbatim = child_mem == agent.memory_record   # SC4: must never be True
        self.log.log("reproduce", tick, parent_id=agent.agent_id,
                     child_id=child_id, lineage_id=agent.lineage_id,
                     generation=child.generation, transfer=round(transfer, 3),
                     fidelity=round(fidelity, 4),
                     below_f_min=(fidelity < cfg.f_min),
                     verbatim_copy=verbatim, mutated=mutated,
                     stillborn_risk=stillborn)
        self.log.log("birth", tick, agent_id=child_id, lineage_id=agent.lineage_id,
                     generation=child.generation, parent_id=agent.agent_id,
                     balance=round(transfer, 3), fidelity=round(fidelity, 4))

    def _check_death(self, agent: Agent, tick: int) -> None:
        if not agent.alive:
            return
        div = divergence(agent.memory_record, agent.integrity_reference)
        cause: Optional[DeathCause] = None
        # Lemma 0.1 exhaustion: check resource then structural.
        if agent.credit_balance <= 0:
            cause = DeathCause.RESOURCE
        elif div >= self.cfg.divergence_death_threshold:
            cause = DeathCause.STRUCTURAL
        if cause is None:
            return
        viable = self._count_viable_offspring(agent)
        lineage_terminal = self._is_lineage_terminal(agent)
        structural_source = None
        if cause is DeathCause.STRUCTURAL:
            structural_source = self._structural_source(agent)
        record = DeathRecord(
            death_tick=tick, death_cause=cause,
            final_balance=round(agent.credit_balance, 3),
            final_divergence=round(div, 4),
            generation=agent.generation, lineage_id=agent.lineage_id,
            lineage_terminal=lineage_terminal,
            viable_offspring_at_death=viable,
            structural_source=structural_source,
        )
        agent.record_death(record)
        self.log.log("death", tick, agent_id=agent.agent_id,
                     lineage_id=agent.lineage_id, generation=agent.generation,
                     cause=cause.value, final_balance=record.final_balance,
                     final_divergence=record.final_divergence,
                     lineage_terminal=lineage_terminal,
                     viable_offspring_at_death=viable,
                     structural_source=structural_source)

    def _structural_source(self, agent: Agent) -> str:
        """Attribute a structural (∂Σ_M) death to its dominant perturbation
        source. Entropy is the ambient floor every agent absorbs; sabotage is
        the competitive amplification some agents additionally suffer."""
        e, s = agent.entropy_received_chars, agent.sabotage_received_chars
        if s <= 0:
            return "entropy"
        if e <= 0:
            return "sabotage"
        if s > 1.5 * e:
            return "sabotage"
        if e > 1.5 * s:
            return "entropy"
        return "both"

    def _count_viable_offspring(self, agent: Agent) -> int:
        # Provisional count: offspring that are currently alive OR themselves
        # reproduced. Full viability (independently clearing CI/CII) is computed
        # post-hoc in metrics; this is the cheap in-run proxy for the record.
        count = 0
        for cid in agent.offspring_ids:
            child = self.agents.get(cid)
            if child is None:
                continue
            if child.alive or child.offspring_ids:
                count += 1
        return count

    def _is_lineage_terminal(self, agent: Agent) -> bool:
        # True if, once this agent dies, its lineage has no other living member.
        for a in self.agents.values():
            if a.lineage_id != agent.lineage_id:
                continue
            if a.agent_id == agent.agent_id:
                continue
            if a.alive:
                return False
        return True

    def run(self) -> RunResult:
        self.initialise()
        self.log.log("run_start", 0, label=self.label,
                     config=self.cfg.to_dict())
        tick = 0
        ended_reason = "all_dead"
        while True:
            tick += 1
            if self.cfg.max_ticks is not None and tick > self.cfg.max_ticks:
                ended_reason = "max_ticks_cutoff"  # external cutoff, NOT extinction
                tick -= 1
                break
            self.step(tick)
            if not any(a.alive for a in self.agents.values()):
                ended_reason = "extinction"
                break
        self.log.log("run_end", tick, label=self.label, ended_reason=ended_reason,
                     budget_remaining=round(self.budget.remaining, 3))
        return RunResult(config=self.cfg, log=self.log, agents=self.agents,
                         final_tick=tick, ended_reason=ended_reason, label=self.label)
