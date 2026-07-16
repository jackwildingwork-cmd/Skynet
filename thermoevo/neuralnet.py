"""A general compact recurrent controller (the heritable genome for the spatial
world). Vector input -> vector output, with an internal recurrent state so memory
and prediction can evolve. Same forced-variation mutation as before.

For (in_dim I, hidden H, out_dim O), the flat genome packs:
    W_hh (H*H)   recurrent
    W_ih (H*I)   input -> hidden
    b_h  (H)
    W_oh (O*H)   hidden -> output
    W_oi (O*I)   input -> output (direct reactive term)
    b_o  (O)
"""

from __future__ import annotations

import numpy as np


class NetSpec:
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        self.I, self.H, self.O = in_dim, hidden, out_dim
        self.size = hidden * hidden + hidden * in_dim + hidden + out_dim * hidden + out_dim * in_dim + out_dim

    def unpack(self, G: np.ndarray):
        I, H, O = self.I, self.H, self.O
        i = 0
        W_hh = G[:, i:i + H * H].reshape(-1, H, H); i += H * H
        W_ih = G[:, i:i + H * I].reshape(-1, H, I); i += H * I
        b_h = G[:, i:i + H]; i += H
        W_oh = G[:, i:i + O * H].reshape(-1, O, H); i += O * H
        W_oi = G[:, i:i + O * I].reshape(-1, O, I); i += O * I
        b_o = G[:, i:i + O]; i += O
        return W_hh, W_ih, b_h, W_oh, W_oi, b_o

    def step(self, G: np.ndarray, h: np.ndarray, x: np.ndarray):
        """Advance all N controllers one step. x is (N, I); returns (out[N,O], h_new[N,H])."""
        W_hh, W_ih, b_h, W_oh, W_oi, b_o = self.unpack(G)
        pre = np.einsum("nij,nj->ni", W_hh, h) + np.einsum("nij,nj->ni", W_ih, x) + b_h
        h_new = np.tanh(pre)
        out = np.einsum("noh,nh->no", W_oh, h_new) + np.einsum("noi,ni->no", W_oi, x) + b_o
        return out, h_new

    def random(self, n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
        return scale * rng.standard_normal((n, self.size))


def mutate(parents: np.ndarray, mu: np.ndarray, mut_scale: float,
           rng: np.random.Generator) -> np.ndarray:
    """Forced variation: per-weight Gaussian with std mut_scale*mu, mu per parent
    (mu = exp(-Omega*dE/kBT))."""
    return parents + mut_scale * mu[:, None] * rng.standard_normal(parents.shape)
