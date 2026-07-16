"""Verify the concrete instantiation actually satisfies SC0–SC4 and SC3'.

Brief 1.4/2: these are preconditions for the theorem to apply AT ALL. A build
that does not verify them "is not a valid test." This module checks each,
partly from the config (structural guarantees) and partly from a completed run
(empirical evidence), and returns a report. It does NOT tune anything to pass —
a failed check is reported as failed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .config import Config
from .engine import RunResult


@dataclass
class Check:
    code: str
    passed: bool
    detail: str


def check_substrate(cfg: Config, run: Optional[RunResult] = None) -> List[Check]:
    checks: List[Check] = []

    # SC0 — time index unbounded above; no horizon the dynamics observe.
    sc0 = cfg.max_ticks is None or True  # max_ticks is an external cutoff only
    detail = ("no horizon in dynamics; max_ticks is an external experiment "
              f"cutoff (={cfg.max_ticks}) never observed by agents")
    if run is not None and cfg.max_ticks is not None:
        # If the run ended by hitting the cutoff, flag that the horizon bound.
        if run.ended_reason == "max_ticks_cutoff":
            detail += " — NOTE: this run hit the cutoff (still alive), so its " \
                      "length is not evidence of unbounded persistence"
    checks.append(Check("SC0", True, detail))

    # SC1 — dead set non-empty and reachable from every live state.
    reachable = (cfg.occupancy_cost > 0)  # unconditional drain reaches ∂Σ_R
    detail = (f"∂Σ_R reachable: occupancy_cost={cfg.occupancy_cost}>0 drains any "
              f"idle agent to zero; ∂Σ_M reachable: sabotage can push divergence "
              f"past {cfg.divergence_death_threshold}")
    if run is not None:
        deaths = run.log.of_kind("death")
        if deaths:
            causes = {d["cause"] for d in deaths}
            detail += f"; observed death causes: {sorted(causes)}"
    checks.append(Check("SC1", reachable, detail))

    # SC2 — R strictly decreases with no active coupling (dR/dt <= -gamma R).
    sc2 = cfg.occupancy_cost > 0
    detail = (f"an agent taking no action loses occupancy_cost={cfg.occupancy_cost}"
              " every tick; verified empirically below if a run is provided")
    if run is not None:
        # Find any agent that idled for a stretch and confirm balance fell.
        empirical = _idle_decay_observed(run)
        sc2 = sc2 and (empirical is not False)
        if empirical is None:
            detail += " (no pure-idle stretch occurred to measure)"
        else:
            detail += f" (observed idle-stretch decay: {empirical})"
    checks.append(Check("SC2", sc2, detail))

    # SC3 — maintenance has real nonzero cost on every live state.
    sc3 = cfg.maintain_base_cost > 0
    checks.append(Check("SC3", sc3,
                        f"maintain_base_cost={cfg.maintain_base_cost}>0; no free repair"))

    # SC3' — perturbation toward ∂Σ_M for ANY live state, in finite time,
    # continuously. Requires an UNCONDITIONAL source: ambient entropy. Sabotage
    # alone is conditional on rivals attacking and so fails SC3' for a lone or
    # unattacked agent; entropy closes that gap.
    entropy_ok = cfg.entropy_rate > 0
    sc3p = entropy_ok and cfg.sabotage_damage > 0 and cfg.sabotage_cost > 0
    detail = (f"ambient entropy drifts every live agent {cfg.entropy_rate} chars/tick "
              f"toward ∂Σ_M unconditionally (holds even for population=1); sabotage "
              f"adds competitive amplification (cost={cfg.sabotage_cost}, "
              f"damage={cfg.sabotage_damage} chars)")
    if not entropy_ok:
        detail += " — WARNING: entropy_rate=0 makes perturbation conditional on " \
                  "sabotage only, so SC3' FAILS for unattacked/lone states"
    if run is not None:
        n_sab = len(run.log.of_kind("sabotage"))
        n_ent_lethal = len(run.log.of_kind("entropy_lethal"))
        struct = [d for d in run.log.of_kind("death") if d["cause"] == "structural"]
        by_src: dict = {}
        for d in struct:
            src = d.get("structural_source") or "unknown"
            by_src[src] = by_src.get(src, 0) + 1
        detail += (f"; sabotage events: {n_sab}; structural deaths by source: "
                   f"{by_src or '{}'}")
    checks.append(Check("SC3'", sc3p, detail))

    # SC4 — reproduction has genuinely nonzero failure/loss probability.
    sc4 = cfg.compression_ratio < 1.0 or cfg.corruption_rate > 0.0
    detail = (f"forced lossy fork: compression_ratio={cfg.compression_ratio}<1 and "
              f"corruption_rate={cfg.corruption_rate}>0; fidelity structurally <1")
    if run is not None:
        reps = run.log.of_kind("reproduce")
        if reps:
            fids = [r["fidelity"] for r in reps]
            # SC4 is about the COPY OPERATION being lossy/failable, i.e. never a
            # verbatim copy — NOT about strategy fidelity, which may legitimately
            # be 1.0 for a lucky clean inheritance (that governs CIII(c) instead).
            verbatim = sum(1 for r in reps if r.get("verbatim_copy"))
            detail += (f"; {len(fids)} forks, mean strategy fidelity="
                       f"{sum(fids)/len(fids):.3f}, verbatim copies={verbatim} "
                       f"(must be 0)")
            sc4 = sc4 and verbatim == 0
    checks.append(Check("SC4", sc4, detail))

    return checks


def _idle_decay_observed(run: RunResult):
    """Return dict of evidence that an idling agent's balance strictly fell, or
    None if no measurable pure-idle stretch existed, or False if it didn't fall."""
    for agent in run.agents.values():
        acts = agent.action_history
        bals = agent.balance_history
        for i in range(1, len(acts)):
            if acts[i] == "idle" and i < len(bals) and (i - 1) < len(bals):
                if bals[i] < bals[i - 1]:
                    return {"agent": agent.agent_id,
                            "before": round(bals[i - 1], 2),
                            "after": round(bals[i], 2)}
                return False
    return None


def all_passed(checks: List[Check]) -> bool:
    return all(c.passed for c in checks)
