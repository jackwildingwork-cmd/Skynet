"""Evolution experiment — does heritable variation + selection change survival?

Tests the hypothesis "evolution is necessary/sufficient for persistence" by
comparing three populations under IDENTICAL substrate conditions (same budget,
same entropy, same tasks, same costs — only the source of strategy variation
differs):

  * control  — fixed non-adaptive baseline policy (always-earn). No strategy.
  * fixed    — adaptive heuristic reading inherited strategy, but reproduction
               only degrades it (corruption channel). Inheritance + selection,
               but NO beneficial variation: evolution cannot improve a lineage.
  * evolving — same heuristic, but each fork also MUTATES the child's strategy
               (heritable variation). Now variation + selection + inheritance =
               genuine Darwinian evolution.

If `evolving` persists (deeper generations, higher R0, longer survival, fewer
extinctions) where `fixed` cannot, that is evidence evolution matters here. If
all three are capped by the same commons-saturation ceiling, evolution does not
rescue persistence and that is reported honestly.

Also measures whether strategies actually MOVE across generations (the direct
signature of selection operating on heritable variation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List

from .config import Config
from .engine import Engine
from .cognition import make_backend
from .memory import parse_strategy, STRATEGY_DEFAULTS
from .metrics import compute_metrics


ARMS = ("control", "fixed", "evolving")


def _arm_config(base: Config, arm: str) -> Config:
    cfg = Config(**base.to_dict())
    cfg.enable_mutation = (arm == "evolving")
    return cfg


def _backend_for(cfg: Config, arm: str):
    return make_backend(cfg, control=(arm == "control"))


@dataclass
class ArmResult:
    arm: str
    seeds: int
    mean_total_survival: float
    mean_max_generation: float
    mean_overall_r0: float
    supercritical_seeds: int
    extinctions: int
    mean_deaths: float
    # mean strategy parameters by generation, pooled across seeds
    strategy_by_generation: Dict[int, Dict[str, float]] = field(default_factory=dict)


def run_arm(base: Config, arm: str, n_seeds: int) -> ArmResult:
    survivals, maxgens, r0s, deaths = [], [], [], []
    supercrit = ext = 0
    strat_accum: Dict[int, Dict[str, List[float]]] = {}

    for seed in range(n_seeds):
        cfg = _arm_config(base, arm)
        cfg.seed = seed
        run = Engine(cfg, _backend_for(cfg, arm), label=arm, seed=seed).run()
        m = compute_metrics(run)
        survivals.append(m.total_lineage_survival)
        maxgens.append(m.max_generation)
        r0s.append(m.overall_r0)
        deaths.append(m.n_deaths)
        supercrit += m.overall_r0 > 1
        ext += run.ended_reason == "extinction"
        # pool strategy params by generation
        for agent in run.agents.values():
            strat = parse_strategy(agent.memory_record)
            bucket = strat_accum.setdefault(agent.generation, {k: [] for k in STRATEGY_DEFAULTS})
            for k in STRATEGY_DEFAULTS:
                bucket[k].append(strat[k])

    strategy_by_gen = {
        gen: {k: mean(v) for k, v in params.items() if v}
        for gen, params in sorted(strat_accum.items())
    }
    return ArmResult(
        arm=arm, seeds=n_seeds,
        mean_total_survival=mean(survivals) if survivals else 0.0,
        mean_max_generation=mean(maxgens) if maxgens else 0.0,
        mean_overall_r0=mean(r0s) if r0s else 0.0,
        supercritical_seeds=supercrit, extinctions=ext,
        mean_deaths=mean(deaths) if deaths else 0.0,
        strategy_by_generation=strategy_by_gen,
    )


def run_experiment(base: Config, n_seeds: int) -> Dict[str, ArmResult]:
    return {arm: run_arm(base, arm, n_seeds) for arm in ARMS}


def format_experiment(results: Dict[str, ArmResult], base: Config) -> str:
    L = [f"=== Evolution experiment: {results['control'].seeds} seeds/arm, "
         f"cutoff={base.max_ticks}, identical substrate ==="]
    L.append("  arm       survival  maxgen   R0     supercrit  extinct  deaths")
    for arm in ARMS:
        r = results[arm]
        L.append(f"  {arm:<9} {r.mean_total_survival:8.0f} {r.mean_max_generation:6.2f} "
                 f"{r.mean_overall_r0:6.3f} {r.supercritical_seeds:>7}/{r.seeds}  "
                 f"{r.extinctions:>5}/{r.seeds}  {r.mean_deaths:5.1f}")

    # Did evolution beat the non-evolving adaptive population?
    fixed, evolving = results["fixed"], results["evolving"]
    L.append("")
    L.append("  -- Does evolution change persistence? (evolving vs fixed) --")
    surv_ratio = (evolving.mean_total_survival / fixed.mean_total_survival
                  if fixed.mean_total_survival else float("inf"))
    L.append(f"     survival: evolving/fixed = {surv_ratio:.2f}")
    L.append(f"     deepest generation: fixed {fixed.mean_max_generation:.2f} "
             f"vs evolving {evolving.mean_max_generation:.2f}")
    L.append(f"     R0: fixed {fixed.mean_overall_r0:.3f} vs evolving {evolving.mean_overall_r0:.3f}")
    L.append(f"     sustained super-criticality (R0>1): fixed "
             f"{fixed.supercritical_seeds}/{fixed.seeds} vs evolving "
             f"{evolving.supercritical_seeds}/{evolving.seeds}")

    # Direct evidence of evolution: strategy drift across generations.
    L.append("")
    L.append("  -- Is selection actually moving strategies? (evolving arm) --")
    sbg = evolving.strategy_by_generation
    gens = sorted(sbg)
    if len(gens) >= 2:
        g0, gN = gens[0], gens[-1]
        L.append(f"     tracked params, gen {g0} -> gen {gN} (pooled means):")
        for k in ("earn_weight", "maintain_weight", "reproduce_weight",
                  "sabotage_weight", "reproduce_threshold", "transfer_fraction"):
            v0 = sbg[g0].get(k, float("nan"))
            vN = sbg[gN].get(k, float("nan"))
            arrow = "->"
            delta = vN - v0
            tag = " (unchanged)" if abs(delta) < 1e-3 else ""
            L.append(f"        {k:<20} {v0:7.3f} {arrow} {vN:7.3f}  (Δ{delta:+.3f}){tag}")
    else:
        L.append("     not enough generations reached to measure drift.")
    return "\n".join(L)
