"""The heritable genome: a compact recurrent controller (the coupling policy).

The genome is a flat weight vector for a tiny RNN that maps the current cue and
its own internal state to a coupling action a(t) and a state update. It is
general enough to evolve anywhere on the spectrum from memoryless-reactive
(ignore the cue, act on a constant) to predictive (integrate cues over time to
anticipate the gradient) — so predictive machinery can *emerge* from selection
rather than being hand-coded.

Variation is thermodynamically forced: offspring weights are perturbed with a
magnitude set by the per-locus copy error rate mu = exp(-Omega*dE/kBT) (paper's
VAR equation). Higher temperature or lower integrity -> more variation, and above
the Eigen threshold the genome cannot be transmitted intact (error catastrophe).

Layout for H hidden units (all rows batched across a population of N agents):
    W_hh (H*H)  recurrent weights
    w_hc (H)    cue -> hidden
    b_h  (H)    hidden bias
    w_a  (H)    hidden -> action
    w_ac (1)    cue -> action (direct reactive term)
    b_a  (1)    action bias
"""

from __future__ import annotations

import numpy as np


def genome_size(H: int) -> int:
    return H * H + 3 * H + 2


def random_genomes(n: int, H: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Small random genomes: generation-1 agents start near-reactive/near-zero,
    so any structure (memory, prediction) is built by evolution, not seeded."""
    return scale * rng.standard_normal((n, genome_size(H)))


def _unpack(G: np.ndarray, H: int):
    i = 0
    W_hh = G[:, i:i + H * H].reshape(-1, H, H); i += H * H
    w_hc = G[:, i:i + H]; i += H
    b_h = G[:, i:i + H]; i += H
    w_a = G[:, i:i + H]; i += H
    w_ac = G[:, i]; i += 1
    b_a = G[:, i]; i += 1
    return W_hh, w_hc, b_h, w_a, w_ac, b_a


def batch_step(G: np.ndarray, h: np.ndarray, cue: float, H: int):
    """Advance every agent's controller one step. Returns (action a[N], h_new[N,H])."""
    W_hh, w_hc, b_h, w_a, w_ac, b_a = _unpack(G, H)
    pre = np.einsum("nij,nj->ni", W_hh, h) + w_hc * cue + b_h
    h_new = np.tanh(pre)
    a = np.einsum("ni,ni->n", w_a, h_new) + w_ac * cue + b_a
    return a, h_new


def mutate(parent: np.ndarray, mu: float, mut_scale: float, rng: np.random.Generator) -> np.ndarray:
    """Forced variation: additive Gaussian on every weight, std = mut_scale * mu,
    where mu = exp(-Omega*dE/kBT) is the thermodynamic per-locus error rate."""
    return parent + mut_scale * mu * rng.standard_normal(parent.shape)


def copy_error_rate(omega: float, dE_copy_max: float, kBT: float) -> float:
    """Per-locus error rate mu = exp(-Omega*dE/kBT) (paper VAR eq)."""
    return float(np.exp(-omega * dE_copy_max / kBT))
