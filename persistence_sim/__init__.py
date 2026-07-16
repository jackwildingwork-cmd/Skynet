"""Persistence Simulation v1 — an empirical test of the CI/CII/CIII framework
in a multi-agent economic substrate.

See the implementation brief and README for the theoretical framing. This
package instantiates the abstract state space (Σ, T, φ, R, M, ρ) in a
computational substrate and reports whether the three-condition structure
appears organically — NOT tuned toward producing an apparent success.
"""

from .config import Config, default_config
from .engine import Engine, RunResult
from .cognition import make_backend
from .metrics import compute_metrics, compare, MetricsReport, Comparison
from .substrate_check import check_substrate, all_passed
from .ensemble import run_ensemble, format_ensemble, EnsembleReport

__all__ = [
    "Config", "default_config", "Engine", "RunResult", "make_backend",
    "compute_metrics", "compare", "MetricsReport", "Comparison",
    "check_substrate", "all_passed",
    "run_ensemble", "format_ensemble", "EnsembleReport",
]
