"""Condition III: loop closure as a branching process, and the extinction
probability that turns "persists indefinitely" into a finite-time-measurable
quantity.

Paper (CIII-a): R0 = Rn(Omega) * phi > 1, where phi is offspring per replication
event. Below R0 = 1 the branching-process extinction theorem gives extinction
with probability 1 as T -> infinity; above it, the extinction probability q < 1
and the lineage survives forever with probability 1 - q > 0.

We model a replication event as phi copy attempts, each yielding a VIABLE
successor with probability equal to the fidelity Rn (an offspring inheriting
enough integrity to itself satisfy CI/CII). Offspring count ~ Binomial(phi, Rn),
so mean = phi*Rn = R0 and the generating function is f(s) = (1 - Rn + Rn*s)^phi.
Extinction probability q is the smallest fixed point of f(q) = q.

This is the crux of the whole test: you never run forever. You measure the
offspring distribution in finite time, obtain q analytically, and INVOKE the
proven theorem to conclude the T -> infinity behaviour. q < 1 with statistical
confidence is the evidence for indefinite (class-level) persistence.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def extinction_probability(phi: int, Rn: float) -> float:
    """Smallest fixed point q of f(q) = (1 - Rn + Rn q)^phi = q.

    q = 1 whenever R0 = phi*Rn <= 1 (sub/critical); q < 1 when R0 > 1."""
    R0 = phi * Rn
    if R0 <= 1.0:
        return 1.0
    q = 0.0
    for _ in range(2000):                      # fixed-point iteration from 0
        q_new = (1.0 - Rn + Rn * q) ** phi
        if abs(q_new - q) < 1e-14:
            break
        q = q_new
    return min(1.0, max(0.0, q))


def basic_reproduction_number(phi: int, Rn: float) -> float:
    return phi * Rn


def simulate_lineage_extinction(phi: int, Rn: float, generations: int,
                                n_lineages: int, rng: np.random.Generator) -> float:
    """Empirical fraction of lineages extinct within `generations`, each started
    from one founder. Offspring ~ Binomial(phi, Rn) per individual per generation."""
    extinct = 0
    for _ in range(n_lineages):
        pop = 1
        for _ in range(generations):
            if pop == 0:
                break
            # total offspring = sum of Binomial(phi, Rn) over `pop` individuals
            pop = int(rng.binomial(phi, Rn, size=pop).sum())
            if pop > 10_000:                   # past this, extinction prob is
                break                          # negligible: treat lineage as survived
        if pop == 0:
            extinct += 1
    return extinct / n_lineages


def error_catastrophe_scan(phi: int, fidelities: np.ndarray) -> np.ndarray:
    """q as a function of fidelity Rn at fixed fecundity phi. q jumps to 1 as Rn
    crosses the Eigen threshold 1/phi from above (the error catastrophe)."""
    return np.array([extinction_probability(phi, float(Rn)) for Rn in fidelities])
