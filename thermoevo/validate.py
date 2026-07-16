"""Milestone-1 validation battery: check the substrate reproduces the paper's
theorems before anything is built on top of it.

Each function returns structured results and a pass/fail, so run_milestone.py can
print them and the tests can assert them. If any of these fail, the substrate is
not trustworthy and the evolutionary layer must not be built on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .constants import Thermo
from .langevin import bm_first_passage_mc, bm_survival_analytic, bm_survival_asymptote
from .integrity import Well, integrity, fidelity, escape_death_prob
from .branching import extinction_probability, simulate_lineage_extinction, basic_reproduction_number
from .environment import Gradient
from .controller import reactive_capture, predictive_capture, predictive_information


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    rows: List[tuple] = field(default_factory=list)


def check_ci_first_passage(rng: np.random.Generator, n: int = 8000) -> Check:
    """CI: survival -> nonzero constant iff net drift mu > 0 (first-passage)."""
    sigma, b = 1.0, 3.0
    rows = []
    ok = True
    for mu in (-0.05, 0.0, 0.05, 0.15):
        _, S = bm_first_passage_mc(mu, sigma, b, t_max=3000, dt=0.5, n=n, rng=rng)
        analytic = bm_survival_asymptote(mu, sigma, b)
        late = float(S[-1])
        rows.append((mu, round(analytic, 3), round(late, 3)))
        if mu > 0:
            ok &= late > 0.05 and abs(late - analytic) < 0.10
        else:
            ok &= late < 0.10
    return Check("CI first-passage (mu>0 -> nonzero survival)", ok,
                 "survival plateau matches 1-exp(-2 mu b/sigma^2) for mu>0; ->0 otherwise", rows)


def check_cii_maintenance_and_gompertz(th: Thermo) -> Check:
    """CII: passive stability dies (Kramers), active survives; and hazard is
    Gompertz (log-hazard linear in age) under net well erosion."""
    rows = []
    # passive vs active survival over a long horizon
    surv = {}
    for label, Eh in (("passive", 0.0), ("active", 0.30)):
        w = Well(depth=th.dV_max, dV_max=th.dV_max)
        p = 1.0
        for _ in range(2000):
            w.maintain(Eh, th, 1.0)
            p *= (1.0 - escape_death_prob([w], th, 1.0))
        surv[label] = p
        rows.append((label, round(w.depth, 2), round(integrity([w]), 2), round(p, 3)))
    # Gompertz: net erosion c per unit time -> B = c/kBT
    w = Well(depth=th.dV_max, dV_max=th.dV_max)
    Eh = 0.20  # repair 0.20 < damage 0.24 -> net erosion 0.04/t
    ages, logh = [], []
    for t in range(280):
        w.maintain(Eh, th, 1.0)
        r = w.escape_rate(th)
        if r > 0:
            ages.append(t); logh.append(np.log(r))
    B_fit = float(np.polyfit(ages, logh, 1)[0])
    B_pred = 0.04 / th.kBT
    rows.append(("Gompertz B (pred 0.040)", round(B_fit, 4), "", ""))
    ok = surv["passive"] < 0.05 and surv["active"] > 0.8 and abs(B_fit - B_pred) < 0.005
    return Check("CII Kramers maintenance + Gompertz emergence", ok,
                 "passive escapes with prob 1; active survives; hazard=A e^{Bt}", rows)


def check_ciii_branching(rng: np.random.Generator, phi: int = 3) -> Check:
    """CIII: empirical lineage extinction q matches the branching theorem, and an
    error catastrophe occurs as fidelity crosses the Eigen threshold 1/phi."""
    rows = []
    ok = True
    for Rn in (0.20, 0.334, 0.45, 0.60, 0.80):
        qa = extinction_probability(phi, Rn)
        qm = simulate_lineage_extinction(phi, Rn, generations=300, n_lineages=4000, rng=rng)
        R0 = basic_reproduction_number(phi, Rn)
        rows.append((Rn, round(R0, 2), round(qa, 3), round(qm, 3)))
        ok &= abs(qa - qm) < 0.05
    # error catastrophe: below 1/phi, q=1; above, q<1
    below = extinction_probability(phi, 1.0 / phi - 0.05)
    above = extinction_probability(phi, 1.0 / phi + 0.05)
    ok &= (below == 1.0) and (above < 1.0)
    return Check("CIII branching q (MC=analytic) + error catastrophe", ok,
                 f"q_MC matches theorem; q=1 below Eigen threshold 1/phi={1/phi:.3f}", rows)


def check_prediction_reward(rng: np.random.Generator, steps: int = 120000) -> Check:
    """The intelligence hook: value of a model (predict - best memoryless) is ~0
    when unpredictable and grows monotonically with predictability rho, tracking
    the predictive information carried."""
    rows = []
    adv = []
    for rho in (0.0, 0.3, 0.6, 0.9, 0.98):
        g = Gradient(rho=rho, cue_noise=0.5)
        theta, cue = g.run(steps, rng)
        fixed = float(g.capture(np.zeros_like(theta), theta).mean())
        cp, xp = predictive_capture(g, theta, cue)
        pred = float(cp.mean())
        Ip = predictive_information(xp[1:], theta[1:])
        rows.append((rho, round(fixed, 3), round(pred, 3), round(pred - fixed, 3), round(Ip, 3)))
        adv.append(pred - fixed)
    # monotone non-decreasing in rho, ~0 at rho=0, clearly positive at rho=0.98
    monotone = all(adv[i + 1] >= adv[i] - 0.01 for i in range(len(adv) - 1))
    ok = abs(adv[0]) < 0.02 and adv[-1] > 0.1 and monotone
    return Check("Prediction reward scales with predictability", ok,
                 "value of internal model ~0 when white, grows with rho, tracks I_pred", rows)


def run_all(seed: int = 0) -> Dict[str, Check]:
    th = Thermo()
    rng = np.random.default_rng(seed)
    return {
        "CI": check_ci_first_passage(rng),
        "CII": check_cii_maintenance_and_gompertz(th),
        "CIII": check_ciii_branching(rng),
        "PRED": check_prediction_reward(rng),
    }
