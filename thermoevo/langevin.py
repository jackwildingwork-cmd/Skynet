"""Condition I: structural stock N_s, its Langevin dynamics, and the
first-passage theorem the paper invokes.

Paper eq (LENS):  dN_s/dt = F_drift(N_s) - gamma_n N_s + sqrt(2 gamma_n k_B T) xi
with an ABSORBING boundary at N_s = 0 (equilibrium / structural death, dSigma_R).
Condition I is  <dN_s/dt + gamma_n N_s>_cycle = <F_drift> > 0.

The paper cites the drift-diffusion first-passage theorem for the net drift mu
of a process with additive noise and an absorbing boundary at 0. Three cases:

  mu < 0 : survival decays exponentially; extinction with probability 1.
  mu = 0 : survival decays as t^{-1/2};   extinction with probability 1.
  mu > 0 : survival -> a NONZERO constant; indefinite persistence is possible,
           with asymptotic survival  S_inf = 1 - exp(-2 mu b / sigma^2),  b = N_s(0).

This module implements the exact analytic survival S(t) for constant-drift
Brownian motion to an absorbing barrier, plus a Monte-Carlo simulator, so the
theorem can be checked rather than assumed. The additive-noise requirement is the
paper's domain condition D2 (timescale separation -> Markovian, additive noise);
with multiplicative noise positive drift would NOT guarantee survival, so we keep
noise additive here deliberately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

_SQRT2 = math.sqrt(2.0)


def _Phi(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF via erf (no scipy)."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / _SQRT2))


def bm_survival_analytic(t: np.ndarray, mu: float, sigma: float, b: float) -> np.ndarray:
    """Exact P(not yet absorbed by time t) for X_t = b + mu t + sigma W_t,
    absorbing barrier at 0, b > 0.  (Reflection-principle result.)"""
    t = np.asarray(t, dtype=float)
    out = np.ones_like(t)
    pos = t > 0
    tt = t[pos]
    s = sigma * np.sqrt(tt)
    term1 = _Phi((b + mu * tt) / s)
    term2 = np.exp(np.clip(-2.0 * mu * b / sigma**2, -700, 700)) * _Phi((-b + mu * tt) / s)
    out[pos] = term1 - term2
    return np.clip(out, 0.0, 1.0)


def bm_survival_asymptote(mu: float, sigma: float, b: float) -> float:
    """S_inf = 1 - exp(-2 mu b / sigma^2) for mu>0, else 0."""
    if mu <= 0:
        return 0.0
    return 1.0 - math.exp(-2.0 * mu * b / sigma**2)


def bm_first_passage_mc(mu: float, sigma: float, b: float, t_max: float,
                        dt: float, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo survival curve for the same process. Returns (times, S(t))."""
    steps = int(round(t_max / dt))
    x = np.full(n, float(b))
    alive = np.ones(n, dtype=bool)
    surv = np.empty(steps + 1)
    surv[0] = 1.0
    sq = sigma * math.sqrt(dt)
    for k in range(1, steps + 1):
        z = rng.standard_normal(n)
        x[alive] += mu * dt + sq * z[alive]
        alive &= x > 0.0
        surv[k] = alive.mean()
    times = np.arange(steps + 1) * dt
    return times, surv


@dataclass
class NsState:
    """Full LENS structural-stock state for an agent (used by the evolutionary
    substrate, not the theorem check). Net active drift F_drift is supplied each
    step by the agent's coupling minus its costs."""
    Ns: float
    gamma_n: float
    kBT: float

    def step(self, F_drift: float, dt: float, rng: np.random.Generator) -> bool:
        """Advance one step; return True if still alive (Ns>0)."""
        noise = math.sqrt(2.0 * self.gamma_n * self.kBT * dt) * rng.standard_normal()
        self.Ns += (F_drift - self.gamma_n * self.Ns) * dt + noise
        if self.Ns <= 0.0:
            self.Ns = 0.0
            return False
        return True

    @property
    def ci_drift_term(self) -> float:
        """The quantity whose cycle average must be > 0 for CI: dNs/dt + gamma_n Ns
        reduces to F_drift; callers average F_drift over a cycle."""
        return 0.0
