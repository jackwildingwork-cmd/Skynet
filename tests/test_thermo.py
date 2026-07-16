"""Tests for the thermoevo thermodynamic core.

These assert that the substrate reproduces the paper's theorems. They are the
gate: the evolutionary layer may only be built on a core that passes these.
"""

import math
import statistics

import numpy as np
import pytest

from thermoevo import (
    Thermo, Well, integrity, fidelity, escape_death_prob, eigen_threshold,
    extinction_probability, basic_reproduction_number, simulate_lineage_extinction,
    bm_survival_asymptote, bm_survival_analytic, bm_first_passage_mc,
    Gradient, predictive_capture, predictive_information,
)
from thermoevo.validate import run_all


# --- CI: first-passage theorem ---------------------------------------------

def test_ci_positive_drift_survives_negative_dies():
    rng = np.random.default_rng(0)
    sigma, b = 1.0, 3.0
    _, S_pos = bm_first_passage_mc(0.15, sigma, b, 2000, 0.5, 6000, rng)
    _, S_neg = bm_first_passage_mc(-0.10, sigma, b, 2000, 0.5, 6000, rng)
    assert S_pos[-1] > 0.1                      # mu>0: nonzero survival
    assert S_neg[-1] < 0.02                     # mu<0: extinction
    # plateau matches analytic asymptote
    assert abs(S_pos[-1] - bm_survival_asymptote(0.15, sigma, b)) < 0.1


def test_ci_analytic_survival_monotone_decreasing():
    t = np.array([1.0, 10.0, 100.0, 1000.0])
    S = bm_survival_analytic(t, 0.1, 1.0, 3.0)
    assert np.all(np.diff(S) <= 1e-9)
    assert S[-1] > bm_survival_asymptote(0.1, 1.0, 3.0) - 1e-6


# --- CII: Kramers maintenance, fidelity, Gompertz ---------------------------

def test_cii_passive_dies_active_survives():
    th = Thermo()
    def survival(Eh):
        w = Well(depth=th.dV_max, dV_max=th.dV_max)
        p = 1.0
        for _ in range(2000):
            w.maintain(Eh, th, 1.0)
            p *= (1.0 - escape_death_prob([w], th, 1.0))
        return p
    assert survival(0.0) < 0.05                 # passive stability fails (Kramers)
    assert survival(0.30) > 0.8                 # active homeostasis survives


def test_cii_gompertz_emerges_from_erosion():
    th = Thermo()
    w = Well(depth=th.dV_max, dV_max=th.dV_max)
    ages, logh = [], []
    for t in range(280):
        w.maintain(0.20, th, 1.0)               # repair 0.20 < damage 0.24
        r = w.escape_rate(th)
        ages.append(t); logh.append(math.log(r))
    B = np.polyfit(ages, logh, 1)[0]
    assert abs(B - 0.04) < 0.005                # log-hazard linear, slope = c/kBT


def test_fidelity_monotone_in_integrity():
    th = Thermo()
    assert fidelity(0.0, th) == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < fidelity(0.5, th) < fidelity(1.0, th) < 1.0


# --- CIII: branching / extinction probability -------------------------------

def test_ciii_extinction_probability_matches_theorem():
    rng = np.random.default_rng(2)
    for Rn in (0.20, 0.45, 0.70):
        qa = extinction_probability(3, Rn)
        qm = simulate_lineage_extinction(3, Rn, 300, 3000, rng)
        assert abs(qa - qm) < 0.05


def test_ciii_error_catastrophe_at_eigen_threshold():
    phi = 3
    thr = eigen_threshold(phi)                  # 1/phi
    assert extinction_probability(phi, thr - 0.05) == 1.0      # sub-threshold: certain extinction
    assert extinction_probability(phi, thr + 0.05) < 1.0       # above: persists with prob 1-q
    assert basic_reproduction_number(phi, thr) == pytest.approx(1.0)


# --- Prediction reward (the intelligence hook) ------------------------------

def test_prediction_pays_only_when_predictable():
    rng = np.random.default_rng(3)
    def value(rho):
        g = Gradient(rho=rho, cue_noise=0.5)
        theta, cue = g.run(60000, rng)
        fixed = g.capture(np.zeros_like(theta), theta).mean()
        cp, _ = predictive_capture(g, theta, cue)
        return cp.mean() - fixed
    v0, v_hi = value(0.0), value(0.95)
    assert abs(v0) < 0.02                        # white noise: model worthless
    assert v_hi > 0.1                            # predictable: model pays
    assert v_hi > v0


def test_predictive_information_zero_when_white():
    rng = np.random.default_rng(4)
    g = Gradient(rho=0.0, cue_noise=0.5)
    theta, cue = g.run(40000, rng)
    _, xp = predictive_capture(g, theta, cue)
    assert predictive_information(xp[1:], theta[1:]) < 0.02


# --- full battery -----------------------------------------------------------

def test_full_battery_passes():
    checks = run_all(seed=0)
    failed = [k for k, c in checks.items() if not c.passed]
    assert not failed, f"failed checks: {failed}"


# --- evolutionary layer -----------------------------------------------------

def test_evolution_persists_via_reproduction():
    from thermoevo.constants import Thermo
    from thermoevo.evolve import EvoConfig, run
    cfg = EvoConfig(th=Thermo(dE_copy_max=4.0), rho=0.9, init_pop=40,
                    ticks=1500, probe_interval=300, seed=0)
    r = run(cfg)
    assert r.ended == "completed" and r.final_pop > 0        # class persists
    gens = [h["mean_gen"] for h in r.history if "mean_gen" in h]
    assert gens[-1] >= 5.0 and gens[-1] > gens[0]            # many generations deep: founders replaced


def test_predictive_coupling_only_when_predictable():
    from thermoevo.constants import Thermo
    from thermoevo.evolve import EvoConfig, run
    def pinfo(rho):
        cfg = EvoConfig(th=Thermo(dE_copy_max=4.0), rho=rho, init_pop=50,
                        ticks=1600, probe_interval=400, seed=1)
        r = run(cfg)
        pr = [h["pred_info"] for h in r.history if "pred_info" in h]
        return statistics.mean(pr[-2:])
    assert pinfo(0.0) < 0.03                                 # white: no prediction possible
    assert pinfo(0.95) > pinfo(0.0)                          # predictable: predictive coupling


# --- spatial foraging world -------------------------------------------------

def test_world_persists_and_forages():
    from thermoevo.world import WorldConfig, run
    r = run(WorldConfig(seed=0, ticks=2000, probe_interval=400))
    assert r.ended == "completed" and r.final_pop > 0        # class persists by foraging
    gens = [h["mean_gen"] for h in r.history if "mean_gen" in h]
    assert gens[-1] >= 4.0                                    # many generations of turnover


def test_foraging_intelligence_evolves_above_random():
    from thermoevo.world import WorldConfig, World, run
    import numpy as np
    # random-genome baseline chemotaxis
    w = World(WorldConfig(seed=7, ticks=10))
    for t in range(1, 40):
        if w.size == 0:
            break
        w.step(t)
    base = float(np.mean(w._chemo))
    r = run(WorldConfig(seed=7, ticks=4000))
    ev = [h["chemotaxis"] for h in r.history if "chemotaxis" in h]
    evolved = statistics.mean(ev[-3:])
    assert evolved > base + 0.1                               # gradient-climbing evolves
