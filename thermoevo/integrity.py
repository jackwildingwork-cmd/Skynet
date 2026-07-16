"""Condition II: homeostatic integrity Omega as a maintained Kramers well, and
the replication fidelity Rn(Omega) that it governs.

Paper: every necessary subsystem j sits in a Kramers well of depth dV_j; thermal
fluctuation escapes it at rate  r = omega0 * exp(-dV_j / k_B T)  (structural death,
dSigma_M). Homeostatic integrity is the weakest well, Omega = min_j dV_j/dV_max.
Active maintenance (Condition II) spends energy E_h from the process's OWN
coupling to hold the wells deep; passive structures (E_h = 0) are dislodged with
probability 1 given enough time (Kramers). Fidelity follows by transition-state
theory (paper eq 15):  Rn(Omega) = 1 - exp(-Omega * dE_copy_max / k_B T).

A consequence the paper derives: with a small net erosion of the well (repair
slightly below damage), dV(t) falls ~linearly, so the hazard
r(t) = omega0 exp(-dV(t)/kBT) = A e^{B t}  is exactly Gompertz-Makeham. Mortality
is therefore NOT bolted on; it emerges from integrity dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from .constants import Thermo


@dataclass
class Well:
    """One structural subsystem's Kramers well."""
    depth: float           # current dV (k_B T units)
    dV_max: float

    def maintain(self, Eh: float, th: Thermo, dt: float) -> None:
        """Damage erodes the well; homeostatic spend Eh repairs it."""
        damage = th.damage_rate * self.dV_max * dt
        repair = th.repair_gain * Eh * dt
        self.depth = min(self.dV_max, max(0.0, self.depth - damage + repair))

    def escape_rate(self, th: Thermo) -> float:
        return th.omega0 * math.exp(-self.depth / th.kBT)


def integrity(wells: List[Well]) -> float:
    """Omega = min normalised well depth across necessary subsystems."""
    return min(w.depth / w.dV_max for w in wells)


def fidelity(omega: float, th: Thermo) -> float:
    """Rn(Omega) from transition-state theory (paper eq 15)."""
    return 1.0 - math.exp(-omega * th.dE_copy_max / th.kBT)


def escape_death_prob(wells: List[Well], th: Thermo, dt: float) -> float:
    """Per-step probability the weakest well is escaped (structural death)."""
    r = max(w.escape_rate(th) for w in wells)   # weakest (deepest escape rate)
    return 1.0 - math.exp(-r * dt)


def eigen_threshold(phi: float) -> float:
    """Minimum fidelity for R0 = Rn*phi > 1: Rn > 1/phi."""
    return 1.0 / phi
