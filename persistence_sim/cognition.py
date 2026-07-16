"""Decision backends — the cognitive substrate (brief 2.1).

HARD CONSTRAINT from the brief: all agents share the same fixed, non-evolving
cognitive capability. Only the inheritable `memory_record` (strategy in text)
varies. Nothing here may let generation number influence how well an agent
decides — if it did, that would be a harness bug leaking capability, not
evolution (brief 2.1, 3).

Backends
--------
* HeuristicBackend  — the "adaptive" mock cognition. Reads the strategy encoded
  in the agent's memory and modulates it by observed LOCAL conditions
  (population density, recent sabotage, remaining budget, own surplus). Fixed
  capability: the decision *rule* is identical for every agent and generation;
  only the strategy TEXT it reads differs. This is what carries selection.
* ControlBackend    — fixed, non-adaptive baseline (always_earn or
  uniform_random) for the class-level persistence comparison (brief 4).
* ClaudeBackend     — real Claude API calls; token cost metered into credits so
  cognition is a genuine metabolic cost (brief 2.2). Same model for everyone,
  so capability is fixed by construction.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from .agent import Agent
from .memory import parse_strategy
from .tasks import Task


class ActionType(Enum):
    EARN = "earn"
    MAINTAIN = "maintain"
    REPRODUCE = "reproduce"
    SABOTAGE = "sabotage"      # a sub-move; still resolves within a tick
    IDLE = "idle"


@dataclass
class Decision:
    action: ActionType
    # MAINTAIN: credits to spend on repair (bounded later by the CII(b) band).
    maintain_spend: float = 0.0
    # REPRODUCE: credits to transfer to the child (on top of the fork cost).
    transfer: float = 0.0
    # SABOTAGE: target agent id.
    sabotage_target: Optional[int] = None
    # Real API token cost of producing this decision (0 for mock backends).
    cognition_cost: float = 0.0


@dataclass
class Observation:
    """What an agent sees before deciding. Deliberately excludes anything that
    could leak capability by generation (e.g. no 'you are gen 7' hint that
    changes reasoning quality)."""
    tick: int
    own_balance: float
    own_divergence: float
    r_min: float
    reserve_surplus: float           # balance - r_min
    population_alive: int
    tasks_available: int
    recent_sabotage_rate: float      # fraction of pop sabotaged last tick
    budget_remaining_frac: float     # global budget left / global budget
    other_agent_ids: List[int]
    strongest_rival_id: Optional[int]
    # Lowest-balance living rival — a "finish the wounded" predation target.
    # Exposing it lets sabotage reach agents too poor to fund repair (the
    # CII(b) reserve-band trap), which is where the structural/∂Σ_M failure
    # mode organically comes from.
    weakest_rival_id: Optional[int]


class Backend:
    def decide(self, agent: Agent, obs: Observation, rng: random.Random) -> Decision:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Adaptive mock cognition.
# ---------------------------------------------------------------------------

class HeuristicBackend(Backend):
    """Fixed decision rule; adapts via strategy text + local conditions."""

    def decide(self, agent: Agent, obs: Observation, rng: random.Random) -> Decision:
        strat = parse_strategy(agent.memory_record)

        # Base action propensities from inherited strategy.
        w = {
            ActionType.EARN: strat["earn_weight"],
            ActionType.MAINTAIN: strat["maintain_weight"],
            ActionType.REPRODUCE: strat["reproduce_weight"],
            ActionType.SABOTAGE: strat["sabotage_weight"],
            ActionType.IDLE: strat["idle_weight"],
        }

        # --- Modulate by local conditions (the "adaptation"). ---
        # 1. If divergence is climbing toward death, prioritise maintenance
        #    (but only preventively while surplus exists — CII(c)).
        danger = obs.own_divergence
        if danger > 0.15:
            w[ActionType.MAINTAIN] += 2.5 * danger
        # 2. Under active sabotage pressure, invest more in defence.
        w[ActionType.MAINTAIN] += 1.5 * obs.recent_sabotage_rate
        # 3. Reproduce only when surplus above reserve clears the threshold.
        if obs.reserve_surplus < strat["reproduce_threshold"]:
            w[ActionType.REPRODUCE] = 0.0
        # 4. Do not reproduce or sabotage when close to the resource floor.
        if obs.reserve_surplus < 0:
            w[ActionType.REPRODUCE] = 0.0
            w[ActionType.SABOTAGE] = 0.0
            w[ActionType.EARN] += 2.0    # scramble to earn back above reserve
        # 5. If tasks are scarce and rivals exist, sabotage becomes more
        #    attractive (competitive pressure), but never when it would breach
        #    reserve.
        if obs.tasks_available <= obs.population_alive and obs.strongest_rival_id is not None:
            if obs.reserve_surplus > 0:
                w[ActionType.SABOTAGE] += 0.6 * strat["sabotage_weight"]
        # 6. Near global budget exhaustion, hunker down: earn/maintain only.
        if obs.budget_remaining_frac < 0.1:
            w[ActionType.REPRODUCE] = 0.0
            w[ActionType.SABOTAGE] = 0.0

        # If nothing affordable/sensible, idle.
        total = sum(w.values())
        if total <= 0:
            return Decision(ActionType.IDLE)

        # Deterministic-given-rng weighted choice.
        action = _weighted_choice(w, rng)

        if action is ActionType.MAINTAIN:
            affordable = max(0.0, obs.reserve_surplus)
            spend = strat["maintain_fraction"] * affordable
            return Decision(ActionType.MAINTAIN, maintain_spend=spend)
        if action is ActionType.REPRODUCE:
            transfer = strat["transfer_fraction"] * obs.reserve_surplus
            return Decision(ActionType.REPRODUCE, transfer=max(0.0, transfer))
        if action is ActionType.SABOTAGE:
            # Predation: prefer a vulnerable rival (removing it frees scarce
            # tasks and can push it past the structural boundary if it is too
            # poor to repair). Fall back to the strongest rival as a threat.
            target = obs.weakest_rival_id or obs.strongest_rival_id
            if strat["sabotage_weight"] > 0.09 and obs.strongest_rival_id is not None:
                # A more aggressive strategy strikes the leading rival instead.
                target = obs.strongest_rival_id
            return Decision(ActionType.SABOTAGE, sabotage_target=target)
        return Decision(action)


def _weighted_choice(weights: Dict[ActionType, float], rng: random.Random) -> ActionType:
    items = [(a, w) for a, w in weights.items() if w > 0]
    total = sum(w for _, w in items)
    r = rng.random() * total
    upto = 0.0
    for a, w in items:
        upto += w
        if r <= upto:
            return a
    return items[-1][0]


# ---------------------------------------------------------------------------
# Fixed, non-adaptive control baseline (brief 4).
# ---------------------------------------------------------------------------

class ControlBackend(Backend):
    def __init__(self, policy: str):
        assert policy in ("always_earn", "uniform_random")
        self.policy = policy

    def decide(self, agent: Agent, obs: Observation, rng: random.Random) -> Decision:
        if self.policy == "always_earn":
            return Decision(ActionType.EARN)
        # uniform_random over the four primary actions (brief 2.4).
        action = rng.choice(
            [ActionType.EARN, ActionType.MAINTAIN, ActionType.REPRODUCE, ActionType.IDLE]
        )
        if action is ActionType.MAINTAIN:
            return Decision(ActionType.MAINTAIN, maintain_spend=max(0.0, obs.reserve_surplus) * 0.5)
        if action is ActionType.REPRODUCE:
            return Decision(ActionType.REPRODUCE, transfer=max(0.0, obs.reserve_surplus) * 0.5)
        return Decision(action)


# ---------------------------------------------------------------------------
# Real Claude cognition. Not invoked by mock runs; kept faithful to brief 2.2.
# ---------------------------------------------------------------------------

class ClaudeBackend(Backend):
    """Calls the Claude API to choose an action; meters token cost into credits.

    A live run using this backend is a real spending decision (brief 5.1) and is
    NOT executed automatically. Requires the `anthropic` package and an API key.
    """

    # Approx public price per 1M tokens (USD) for the default small model.
    _PRICE_IN = 1.0 / 1_000_000
    _PRICE_OUT = 5.0 / 1_000_000

    def __init__(self, model: str, usd_per_credit: float, max_tokens: int):
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - only on live runs
            raise RuntimeError(
                "ClaudeBackend requires the 'anthropic' package. Install it and "
                "provide an API key to run the live cognitive substrate."
            ) from exc
        import anthropic
        self._client = anthropic.Anthropic()
        self._model = model
        self._usd_per_credit = usd_per_credit
        self._max_tokens = max_tokens

    def _system_prompt(self) -> str:
        return (
            "You are an autonomous agent in a resource-constrained economy. "
            "Each tick you pick exactly one action to persist as long as your "
            "lineage can. Actions: earn (attempt a task for credits), maintain "
            "(spend credits to repair your memory integrity), reproduce (fork a "
            "child, transferring credits), sabotage (damage a rival's "
            "integrity), idle. Respond ONLY with a JSON object: "
            '{"action": "...", "maintain_spend": n, "transfer": n, '
            '"sabotage_target": id_or_null}. Your inherited strategy notes and '
            "current situation follow."
        )

    def decide(self, agent: Agent, obs: Observation, rng: random.Random) -> Decision:  # pragma: no cover
        import anthropic  # noqa: F401
        user = json.dumps({
            "memory_record": agent.memory_record,
            "balance": round(agent.credit_balance, 2),
            "divergence": round(obs.own_divergence, 3),
            "reserve_surplus": round(obs.reserve_surplus, 2),
            "population_alive": obs.population_alive,
            "tasks_available": obs.tasks_available,
            "recent_sabotage_rate": round(obs.recent_sabotage_rate, 3),
            "budget_remaining_frac": round(obs.budget_remaining_frac, 3),
            "rival_id": obs.strongest_rival_id,
        })
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system_prompt(),
            messages=[{"role": "user", "content": user}],
        )
        usage = resp.usage
        usd = usage.input_tokens * self._PRICE_IN + usage.output_tokens * self._PRICE_OUT
        cognition_cost = usd / self._usd_per_credit

        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        decision = self._parse(text, obs)
        decision.cognition_cost = cognition_cost
        return decision

    @staticmethod
    def _parse(text: str, obs: Observation) -> Decision:  # pragma: no cover
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return Decision(ActionType.IDLE)
        try:
            action = ActionType(data.get("action", "idle"))
        except ValueError:
            action = ActionType.IDLE
        return Decision(
            action=action,
            maintain_spend=float(data.get("maintain_spend") or 0.0),
            transfer=float(data.get("transfer") or 0.0),
            sabotage_target=data.get("sabotage_target"),
        )


def make_backend(cfg, *, control: bool = False) -> Backend:
    if control:
        return ControlBackend(cfg.control_policy)
    if cfg.cognition_backend == "claude":
        return ClaudeBackend(cfg.claude_model, cfg.usd_per_credit, cfg.max_decision_tokens)
    return HeuristicBackend()
