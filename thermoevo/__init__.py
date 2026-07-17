"""thermoevo — evolution from thermodynamic first principles.

A substrate built on the Persistence Theorem (Wilding): structural stock N_s under
Langevin first-passage dynamics (CI), homeostatic integrity Omega as maintained
Kramers wells governing replication fidelity Rn (CII), and loop closure as a
branching process whose extinction probability q makes "indefinite persistence"
a finite-time-measurable quantity (CIII). Variation is thermodynamically forced
(Rn<1 at T>0); selection follows from a finite shared gradient; evolution is the
consequence. The environment can reward prediction: an internal model captures
more free energy in proportion to the environment's predictability.

Milestone 1 (this commit) builds and validates the thermodynamic core against the
paper's theorems. The evolutionary layer is built on top only once the core is
trustworthy.
"""

from .constants import Thermo, DEFAULT
from .langevin import (
    NsState, bm_survival_analytic, bm_survival_asymptote, bm_first_passage_mc,
)
from .integrity import Well, integrity, fidelity, escape_death_prob, eigen_threshold
from .branching import (
    extinction_probability, basic_reproduction_number, simulate_lineage_extinction,
)
from .environment import Gradient
from .controller import reactive_capture, predictive_capture, predictive_information
from .validate import run_all, Check
from .field import ResourceField
from .producers import Producers, ProducerConfig
from .neuralnet import NetSpec
from .world import WorldConfig, World, run as run_world

__all__ = [
    "Thermo", "DEFAULT",
    "NsState", "bm_survival_analytic", "bm_survival_asymptote", "bm_first_passage_mc",
    "Well", "integrity", "fidelity", "escape_death_prob", "eigen_threshold",
    "extinction_probability", "basic_reproduction_number", "simulate_lineage_extinction",
    "Gradient", "reactive_capture", "predictive_capture", "predictive_information",
    "run_all", "Check",
    "ResourceField", "Producers", "ProducerConfig",
    "NetSpec", "WorldConfig", "World", "run_world",
]
