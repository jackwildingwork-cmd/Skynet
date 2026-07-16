"""Dimensionless thermodynamic constants for the substrate.

Everything is in units of the thermal energy k_B T (so k_B = 1 and temperature T
is the control knob that sets both the noise amplitude and, through it, the
forced replication error rate). These are defaults; experiments override them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thermo:
    kB: float = 1.0
    T: float = 1.0                 # temperature (the control knob)
    # Structural-stock (Ns) Langevin dynamics
    gamma_n: float = 0.05          # passive structural relaxation rate
    # Kramers-well / integrity (Omega) dynamics
    dV_max: float = 12.0           # maximum well depth (k_B T units)
    omega0: float = 1.0            # attempt frequency (Kramers prefactor)
    damage_rate: float = 0.02      # thermal damage rate on well depth per unit time
    repair_gain: float = 1.0       # well depth restored per unit homeostatic spend
    # Replication fidelity (transition-state theory)
    dE_copy_max: float = 6.0       # max copy-discrimination energy (k_B T units)

    @property
    def kBT(self) -> float:
        return self.kB * self.T


DEFAULT = Thermo()
