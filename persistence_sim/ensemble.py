"""Ensemble runner — because single-seed runs are misleading.

A single run's outcome (deaths, signatures, R0, adaptive-vs-control) varies a
lot with the seed. Reporting one seed risks exactly the "manufacture a
favorable-looking result" failure the brief warns against. This module runs the
adaptive and control populations across many seeds and reports the DISTRIBUTION,
so conclusions rest on the ensemble, not a lucky draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Dict, List

from .config import Config
from .engine import Engine
from .cognition import make_backend
from .metrics import compute_metrics, compare


@dataclass
class EnsembleReport:
    n_seeds: int
    # per-seed adaptive summaries
    deaths: List[int]
    both_death_modes_seeds: int
    resource_deaths: int
    structural_deaths: int
    structural_source_counts: Dict[str, int]
    mean_signatures: float
    signature_presence: Dict[str, int]     # per signature: seeds it appeared in
    extinctions: int
    mean_max_generation: float
    mean_overall_r0: float
    supercritical_seeds: int               # seeds with overall R0 > 1
    # adaptive vs control
    outlast_ratios: List[float]
    adaptive_outlasts_seeds: int
    mean_ratio: float
    median_ratio: float


def run_ensemble(cfg: Config, n_seeds: int) -> EnsembleReport:
    deaths, ratios, r0s, maxgens = [], [], [], []
    both = ext = struct = res = supercrit = outlast = 0
    src_counts: Dict[str, int] = {}
    sig_presence: Dict[str, int] = {}
    sig_totals: List[int] = []

    for seed in range(n_seeds):
        a = Engine(cfg, make_backend(cfg), label="adaptive", seed=seed).run()
        c = Engine(cfg, make_backend(cfg, control=True), label="control", seed=seed).run()
        ma, mc = compute_metrics(a), compute_metrics(c)
        cmp = compare(ma, mc)

        deaths.append(ma.n_deaths)
        cc = ma.death_cause_counts
        both += len(cc) >= 2
        struct += cc.get("structural", 0)
        res += cc.get("resource", 0)
        ext += a.ended_reason == "extinction"
        maxgens.append(ma.max_generation)
        r0s.append(ma.overall_r0)
        supercrit += ma.overall_r0 > 1
        ratios.append(cmp.ratio)
        outlast += cmp.adaptive_outlasts_control

        present = 0
        for k, v in ma.pairwise_signatures.items():
            if v > 0:
                sig_presence[k] = sig_presence.get(k, 0) + 1
                present += 1
        sig_totals.append(present)
        for d in a.log.of_kind("death"):
            if d["cause"] == "structural":
                s = d.get("structural_source") or "unknown"
                src_counts[s] = src_counts.get(s, 0) + 1

    return EnsembleReport(
        n_seeds=n_seeds, deaths=deaths, both_death_modes_seeds=both,
        resource_deaths=res, structural_deaths=struct,
        structural_source_counts=src_counts,
        mean_signatures=mean(sig_totals) if sig_totals else 0.0,
        signature_presence=sig_presence, extinctions=ext,
        mean_max_generation=mean(maxgens) if maxgens else 0.0,
        mean_overall_r0=mean(r0s) if r0s else 0.0,
        supercritical_seeds=supercrit,
        outlast_ratios=ratios, adaptive_outlasts_seeds=outlast,
        mean_ratio=mean(ratios) if ratios else 0.0,
        median_ratio=median(ratios) if ratios else 0.0,
    )


def format_ensemble(r: EnsembleReport) -> str:
    L = [f"=== Ensemble over {r.n_seeds} seeds (adaptive vs control) ==="]
    L.append("  -- mortality --")
    L.append(f"     mean deaths/run: {mean(r.deaths):.1f}; total resource={r.resource_deaths}, "
             f"structural={r.structural_deaths}")
    L.append(f"     structural death sources: {r.structural_source_counts or '{}'}")
    L.append(f"     runs with BOTH death modes present: {r.both_death_modes_seeds}/{r.n_seeds}")
    L.append(f"     full-population extinctions: {r.extinctions}/{r.n_seeds}")
    L.append("  -- CIII (loop closure) --")
    L.append(f"     mean deepest generation: {r.mean_max_generation:.1f}")
    L.append(f"     mean overall R0 (viable offspring/parent): {r.mean_overall_r0:.3f} "
             f"({'>1' if r.mean_overall_r0 > 1 else '<=1 — sub-critical, extinction certain in unbounded time (CIII(b))'})")
    L.append(f"     seeds achieving sustained R0>1: {r.supercritical_seeds}/{r.n_seeds}")
    L.append("  -- Falsification: six pairwise signatures (seeds each appeared in) --")
    for k in ("CI_without_CII", "CI_without_CIII", "CII_without_CI",
              "CII_without_CIII", "CIII_without_CI", "CIII_without_CII"):
        L.append(f"     {k}: {r.signature_presence.get(k, 0)}/{r.n_seeds}")
    L.append(f"     mean distinct signatures per run: {r.mean_signatures:.2f}/6")
    L.append("  -- Class-level persistence: adaptive vs control --")
    L.append(f"     adaptive/control survival ratio: mean {r.mean_ratio:.2f}, "
             f"median {r.median_ratio:.2f}")
    L.append(f"     adaptive meaningfully outlasts control (ratio>=1.15): "
             f"{r.adaptive_outlasts_seeds}/{r.n_seeds} seeds")
    if r.adaptive_outlasts_seeds > r.n_seeds * 0.6:
        L.append("     -> selection pressure IS operating: the adaptive policy "
                 "(which funds maintenance/CII) robustly outlives the non-adaptive "
                 "baseline that does not.")
    else:
        L.append("     -> selection pressure NOT robustly demonstrated across seeds.")
    return "\n".join(L)
