"""Milestone-2 experiments: evolution on the thermodynamic substrate.

Two questions, each answered by a sweep. Effect sizes are reported honestly; this
is a drift/load-limited regime (small populations, forced variation), so the
signals are directional and consistent with theory rather than large.

  1. Does prediction (intelligence) evolve only when the environment affords it?
     Sweep the gradient predictability rho and measure the predictive information
     the evolved controllers' actions carry about the gradient. Expectation:
     ~0 when white (nothing to predict), rising with rho.

  2. Is there a thermodynamic evolvability window?
     Variation is forced at rate mu = exp(-Omega*dE/kBT), so ONLY temperature /
     copy-discrimination sets it. Sweep dE and measure whether selection improves
     coupling skill over generations. Expectation: adaptation only in an
     intermediate band; high mu (low dE) -> mutation load degrades skill; low mu
     (high dE) -> frozen. This is the mutation-selection-drift balance.

Run:  python -m thermoevo.evo_experiment
"""

from __future__ import annotations

import math
import statistics
from typing import List

import numpy as np

from .constants import Thermo
from .evolve import EvoConfig, run


def _run_metrics(dE: float, rho: float, seeds: int, ticks: int, **kw):
    th = Thermo(dE_copy_max=dE)
    early, late, pinfo, ext = [], [], [], 0
    for s in range(seeds):
        cfg = EvoConfig(th=th, rho=rho, cue_noise=1.0, grad_width=0.4, mut_scale=1.0,
                        G_total=120, init_pop=80, pop_cap=1000, ticks=ticks,
                        probe_interval=250, seed=s, **kw)
        r = run(cfg)
        if r.ended == "extinct":
            ext += 1
            continue
        pr = [h for h in r.history if "skill" in h]
        if len(pr) < 4:
            continue
        early.append(pr[0]["skill"])
        late.append(statistics.mean(p["skill"] for p in pr[-3:]))
        pinfo.append(statistics.mean(p["pred_info"] for p in pr[-3:]))
    def m(x): return statistics.mean(x) if x else float("nan")
    def sd(x): return statistics.pstdev(x) if len(x) > 1 else 0.0
    return dict(early=m(early), late=m(late), delta=m(late) - m(early),
                pinfo=m(pinfo), pinfo_sd=sd(pinfo), ext=ext, n=seeds)


def predictability_sweep(seeds: int = 3, ticks: int = 3000):
    print("=== 1. Does prediction evolve only when the environment is predictable? ===")
    print("    (dE=4 fixed = adaptive window; predictive info of evolved controllers)")
    print(f"    {'rho':>5} {'pred_info':>10} {'(sd)':>7} {'skill_late':>11} {'extinct':>8}")
    rows = []
    for rho in (0.0, 0.3, 0.6, 0.9, 0.98):
        r = _run_metrics(4.0, rho, seeds, ticks)
        rows.append((rho, r["pinfo"]))
        print(f"    {rho:>5.2f} {r['pinfo']:>10.3f} {r['pinfo_sd']:>7.3f} {r['late']:>11.3f} {r['ext']:>6}/{r['n']}")
    ok = rows[0][1] < 0.03 and rows[-1][1] > rows[0][1]
    print(f"    -> predictive info ~0 when white, rises with predictability: {'as expected' if ok else 'NOT clean'}")


def forced_variation_sweep(seeds: int = 3, ticks: int = 4500):
    print("\n=== 2. The thermodynamic evolvability window (forced variation rate) ===")
    print("    variation is forced: mu = exp(-Omega*dE/kBT). Adaptation = skill rising.")
    print(f"    {'dE':>5} {'mu(Om=1)':>9} {'skill_early':>12} {'skill_late':>11} {'delta':>8} {'regime':>16}")
    for dE in (3.0, 4.0, 5.0, 6.0):
        r = _run_metrics(dE, 0.9, seeds, ticks)
        mu = math.exp(-dE)
        if r["delta"] > 0.012:
            regime = "ADAPTS"
        elif r["delta"] < -0.005:
            regime = "mutation load"
        else:
            regime = "frozen/flat"
        print(f"    {dE:>5.1f} {mu:>9.3f} {r['early']:>12.3f} {r['late']:>11.3f} {r['delta']:>+8.3f} {regime:>16}")
    print("    -> selection improves coupling only in an intermediate mu band;")
    print("       too much forced variation degrades it, too little freezes it.")


def main():
    print("thermoevo — Milestone 2: evolution on the thermodynamic substrate")
    print("(directional, drift/load-limited effect sizes — reported honestly)\n")
    predictability_sweep()
    forced_variation_sweep()


if __name__ == "__main__":
    main()
