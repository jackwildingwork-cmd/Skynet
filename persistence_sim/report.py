"""Human-readable reporting of a run, faithful to the brief's epistemic stance.

Reports negative / inconclusive results as such (brief 6). Does not reframe an
extinction or an identical-death population as a partial success.
"""

from __future__ import annotations

from typing import List

from .metrics import Comparison, MetricsReport
from .substrate_check import Check, all_passed


def format_substrate(checks: List[Check]) -> str:
    lines = ["=== Substrate conditions (preconditions for the theorem) ==="]
    for c in checks:
        mark = "PASS" if c.passed else "FAIL"
        lines.append(f"  [{mark}] {c.code}: {c.detail}")
    lines.append(f"  -> substrate {'valid' if all_passed(checks) else 'INVALID — results not a valid test'}")
    return "\n".join(lines)


def format_metrics(m: MetricsReport) -> str:
    L = [f"=== Metrics: {m.label} population ==="]
    L.append(f"  ended: {m.ended_reason} at tick {m.final_tick}; "
             f"agents ever alive: {m.n_agents}; deaths: {m.n_deaths}")
    L.append(f"  death causes: {m.death_cause_counts or '{}'}")
    if m.ended_reason == "extinction":
        L.append("  NOTE: population went EXTINCT — the class-level claim (q<1) "
                 "is NOT demonstrated by this run.")
    if m.ended_reason == "max_ticks_cutoff":
        L.append("  NOTE: run hit the external tick cutoff with survivors; length "
                 "is bounded by the experiment, not by the dynamics (SC0).")

    L.append("  -- CI (gradient coupling) --")
    L.append(f"     lineages with positive cycle-averaged <dR/dt+gammaR>: "
             f"{m.ci_lineage_positive_frac:.2f}; stable (not a lucky spike): "
             f"{m.ci_stable_frac:.2f}")

    L.append("  -- CII (active self-funded homeostasis) --")
    L.append(f"     integrity/structural failure rate: {m.integrity_failure_rate:.2f} "
             f"of deaths; maintenance-denied (hit CII(b) reserve band) events: "
             f"{m.maintenance_denied_events}")
    L.append(f"     of agents that maintained: preventive {m.preventive_frac:.2f}, "
             f"reactive-only {m.reactive_only_frac:.2f} "
             f"(CII(c) wants preventive margin, not reactive-only)")

    L.append("  -- CIII (loop closure) --")
    r0 = ", ".join(f"g{g}:{v:.2f}" for g, v in sorted(m.r0_by_generation.items()))
    L.append(f"     R0 (viable offspring/parent) by generation: {r0 or 'n/a'}")
    L.append(f"     overall R0: {m.overall_r0:.3f} "
             f"({'>1 — above criticality' if m.overall_r0 > 1 else '<=1 — BELOW criticality, extinction certain (CIII(b))'})")
    L.append(f"     deepest generation reached: {m.max_generation}")

    L.append("  -- Class-level persistence --")
    L.append(f"     total lineage survival: {m.total_lineage_survival} tick-lineages; "
             f"mean per lineage: {m.mean_lineage_survival:.1f}")

    L.append("  -- Falsification: six pairwise-independence signatures (Lemma 0.2) --")
    present = [k for k, v in m.pairwise_signatures.items() if v > 0]
    for k, v in m.pairwise_signatures.items():
        ex = m.signature_examples.get(k, [])
        tag = "" if v else "   (absent)"
        L.append(f"     {k}: {v}{tag}" + (f"  e.g. agents {ex}" if ex else ""))
    L.append(f"     -> {len(present)}/6 signatures appeared organically.")
    if m.identical_death_warning:
        L.append("     WARNING: every death has the SAME cause — evidence the "
                 "three-condition structure is NOT being exercised. Treat as "
                 "negative/inconclusive, not partial success.")
    elif len(present) >= 3:
        L.append("     Multiple distinct signatures present — consistent with a "
                 "faithful instantiation exercising the structure.")
    else:
        L.append("     Few signatures present — inconclusive on faithfulness.")
    return "\n".join(L)


def format_comparison(cmp: Comparison) -> str:
    L = ["=== Class-level persistence: adaptive vs control ==="]
    L.append(f"  adaptive total lineage survival: {cmp.adaptive_total_survival} "
             f"(mean {cmp.adaptive_mean_survival:.1f}, max gen {cmp.adaptive_max_generation})")
    L.append(f"  control  total lineage survival: {cmp.control_total_survival} "
             f"(mean {cmp.control_mean_survival:.1f}, max gen {cmp.control_max_generation})")
    L.append(f"  ratio adaptive/control: {cmp.ratio:.2f}")
    if cmp.adaptive_outlasts_control:
        L.append("  -> adaptive population MEANINGFULLY outlasts control: selection "
                 "pressure is operating.")
    else:
        L.append("  -> adaptive population does NOT meaningfully outlast control. "
                 "Per brief 4, selection pressure is NOT demonstrated, regardless "
                 "of other results.")
    return "\n".join(L)
