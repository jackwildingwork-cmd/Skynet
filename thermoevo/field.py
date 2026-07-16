"""The environment: a discrete, patchy, replenishing free-energy field.

Not one massive shared cache and NOT scaled to the population — a fixed set of
discrete fertile patches on a torus, each regrowing logistically toward a capped
capacity. Total environmental capacity is fixed by the world, so more agents means
less each (competition) and depletion is real. Light diffusion spreads a little
resource around each patch, creating a local gradient an agent can sense and climb
— so "finding" rewards both luck (stumbling on a patch) and intellect (sensing and
climbing the gradient, remembering where patches are).
"""

from __future__ import annotations

import numpy as np


class ResourceField:
    def __init__(self, W: int, n_patches: int, capacity: float,
                 grow_rate: float, diffuse: float, seed_flux: float,
                 rng: np.random.Generator):
        self.W = W
        self.capacity = capacity
        self.grow_rate = grow_rate
        self.diffuse = diffuse
        self.seed_flux = seed_flux
        # fertile sites (fixed locations) — the discrete gradient sources
        self.fertile = np.zeros((W, W), dtype=bool)
        idx = rng.choice(W * W, size=n_patches, replace=False)
        self.fertile.flat[idx] = True
        self.R = np.zeros((W, W))
        self.R[self.fertile] = capacity                    # start full

    def total(self) -> float:
        return float(self.R.sum())

    def regrow(self) -> None:
        # logistic regrowth at fertile sites (+ a small seed so a depleted patch
        # can recover from zero); barren sites only receive diffusion.
        f = self.fertile
        self.R[f] += (self.grow_rate * self.R[f] * (1.0 - self.R[f] / self.capacity)
                      + self.seed_flux)
        # diffusion (4-neighbour) to create climbable gradients around patches
        if self.diffuse > 0:
            R = self.R
            lap = (np.roll(R, 1, 0) + np.roll(R, -1, 0) +
                   np.roll(R, 1, 1) + np.roll(R, -1, 1) - 4.0 * R)
            self.R = R + self.diffuse * lap
        np.clip(self.R, 0.0, self.capacity, out=self.R)

    def sense(self, ix: np.ndarray, iy: np.ndarray):
        """Per-agent local sensing: [R_here, grad_x, grad_y] at integer cells."""
        W = self.W
        here = self.R[ix, iy]
        gx = self.R[(ix + 1) % W, iy] - self.R[(ix - 1) % W, iy]
        gy = self.R[ix, (iy + 1) % W] - self.R[ix, (iy - 1) % W]
        return here, gx, gy

    def consume(self, ix: np.ndarray, iy: np.ndarray, demand: np.ndarray) -> np.ndarray:
        """Agents at cells (ix,iy) each want `demand` energy. Where several share a
        cell they compete (scramble): the cell's resource is split in proportion to
        demand. Returns the amount each agent actually gets, and depletes the field.
        """
        W = self.W
        flat = ix * W + iy
        # total demand per cell
        cell_demand = np.zeros(W * W)
        np.add.at(cell_demand, flat, demand)
        avail = self.R.flat[flat]                          # resource at each agent's cell
        cd = cell_demand[flat]
        share = np.where(cd > 0, np.minimum(1.0, avail / cd), 0.0)
        got = demand * share
        # deplete field
        taken = np.zeros(W * W)
        np.add.at(taken, flat, got)
        self.R.flat[:] = np.maximum(0.0, self.R.flat[:] - taken)
        return got
