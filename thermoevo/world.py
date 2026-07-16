"""Spatial evolutionary world: agents forage a discrete, replenishing gradient
field, keeping the validated thermodynamic core (N_s Langevin + absorbing
boundary, Omega Kramers wells, forced-error reproduction).

Each tick an agent senses its local resource gradient and its own hunger, its
evolved recurrent controller outputs a move, it pays a search cost proportional to
the move plus basal + homeostatic costs, and consumes what it finds at its new
cell (competing with any agents sharing it). Condition I is explicit and local:
intake must exceed the cost of finding and consuming, or N_s drifts to the
absorbing boundary and the agent dies. Selection acts on nothing but that.

What can matter here, and is measured: intellect (moving up-gradient / using memory
to forage well), luck (stochastic patch encounters), adaptability, and
competition (shared patches deplete). Cooperation is not built in; if it appears it
is emergent.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, List

import numpy as np

from .constants import Thermo
from .field import ResourceField
from .neuralnet import NetSpec, mutate


@dataclass
class WorldConfig:
    th: Thermo = dc_field(default_factory=lambda: Thermo(dE_copy_max=4.0))
    # field — discrete fertile patches, capped, replenishing (not pop-scaled)
    W: int = 48
    n_patches: int = 120
    capacity: float = 80.0
    grow_rate: float = 0.25
    diffuse: float = 0.22
    seed_flux: float = 0.8
    # agent / controller
    hidden: int = 5
    init_pop: int = 60
    init_scale: float = 0.5
    pop_cap: int = 800
    max_step: float = 1.5
    max_intake: float = 8.0
    # energetics (Condition I: intake must beat the cost of finding + consuming).
    # Scale is set well above the thermal noise sqrt(2 gamma_n kBT) so structural
    # stock is not random-walked into the absorbing boundary.
    occupancy: float = 0.5
    move_cost: float = 0.4           # per unit distance moved (cost of finding)
    homeostasis_spend: float = 0.5
    gamma_n: float = 0.05
    dt: float = 1.0
    # reproduction
    repro_threshold: float = 30.0
    repro_transfer: float = 12.0
    landauer_cost: float = 2.0
    mut_scale: float = 1.0
    child_scatter: float = 2.0
    # run
    ticks: int = 4000
    probe_interval: int = 200
    seed: int = 0


IN_DIM, OUT_DIM = 4, 2   # sense [R_here, grad_x, grad_y, hunger] -> move [vx, vy]


class World:
    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.net = NetSpec(IN_DIM, cfg.hidden, OUT_DIM)
        self.field = ResourceField(cfg.W, cfg.n_patches, cfg.capacity, cfg.grow_rate,
                                   cfg.diffuse, cfg.seed_flux, self.rng)
        n = cfg.init_pop
        self.G = self.net.random(n, cfg.init_scale, self.rng)
        self.h = np.zeros((n, cfg.hidden))
        # spawn founders ON fertile patches (organisms arise where the gradient is),
        # jittered, so generation 1 has a fighting chance to forage and reproduce.
        fx, fy = np.nonzero(self.field.fertile)
        pick = self.rng.integers(0, len(fx), size=n)
        self.pos = (np.stack([fx[pick], fy[pick]], axis=1).astype(float)
                    + self.rng.standard_normal((n, 2))) % cfg.W
        self.Ns = np.full(n, cfg.repro_threshold * 0.6)
        self.depth = np.full(n, cfg.th.dV_max)
        self.generation = np.ones(n, dtype=int)
        self._chemo = []   # rolling chemotaxis alignment samples

    @property
    def size(self) -> int:
        return self.G.shape[0]

    def _cells(self, pos):
        W = self.cfg.W
        ix = np.floor(pos[:, 0]).astype(int) % W
        iy = np.floor(pos[:, 1]).astype(int) % W
        return ix, iy

    def step(self, t: int) -> None:
        cfg = self.cfg
        if self.size == 0:
            return
        self.field.regrow()
        ix, iy = self._cells(self.pos)
        here, gx, gy = self.field.sense(ix, iy)
        hunger = np.clip(1.0 - self.Ns / cfg.repro_threshold, 0.0, 1.0)
        x = np.stack([here / cfg.capacity, gx, gy, hunger], axis=1)

        out, self.h = self.net.step(self.G, self.h, x)
        move = cfg.max_step * np.tanh(out)             # (N,2)
        step_len = np.linalg.norm(move, axis=1)
        # chemotaxis alignment: does the move climb the sensed gradient? (intellect)
        grad = np.stack([gx, gy], axis=1)
        gnorm = np.linalg.norm(grad, axis=1)
        good = (gnorm > 1e-6) & (step_len > 1e-6)
        if good.any():
            align = (move[good] * grad[good]).sum(1) / (step_len[good] * gnorm[good])
            self._chemo.append(float(align.mean()))

        self.pos = (self.pos + move) % cfg.W
        nix, niy = self._cells(self.pos)
        intake = self.field.consume(nix, niy, np.full(self.size, cfg.max_intake))

        move_cost = cfg.move_cost * step_len
        F_drift = intake - cfg.occupancy - move_cost - cfg.homeostasis_spend
        noise = np.sqrt(2.0 * cfg.gamma_n * cfg.th.kBT * cfg.dt) * self.rng.standard_normal(self.size)
        self.Ns = self.Ns + (F_drift - cfg.gamma_n * self.Ns) * cfg.dt + noise

        # integrity maintenance + Kramers escape (structural death)
        damage = cfg.th.damage_rate * cfg.th.dV_max * cfg.dt
        repair = cfg.th.repair_gain * cfg.homeostasis_spend * cfg.dt
        self.depth = np.clip(self.depth - damage + repair, 0.0, cfg.th.dV_max)
        p_escape = 1.0 - np.exp(-cfg.th.omega0 * np.exp(-self.depth / cfg.th.kBT) * cfg.dt)
        escaped = self.rng.random(self.size) < p_escape

        alive = (self.Ns > 0.0) & (~escaped)
        self._compact(alive)
        if self.size == 0:
            return
        can = (self.Ns > cfg.repro_threshold) & (self.Ns > cfg.repro_transfer + cfg.landauer_cost)
        idx = np.nonzero(can)[0]
        if idx.size:
            self._reproduce(idx)

    def _compact(self, alive):
        self.G = self.G[alive]; self.h = self.h[alive]; self.pos = self.pos[alive]
        self.Ns = self.Ns[alive]; self.depth = self.depth[alive]
        self.generation = self.generation[alive]

    def _reproduce(self, idx):
        cfg = self.cfg
        omega = self.depth[idx] / cfg.th.dV_max
        mu = np.exp(-omega * cfg.th.dE_copy_max / cfg.th.kBT)       # forced error rate
        children = mutate(self.G[idx], mu, cfg.mut_scale, self.rng)
        self.Ns[idx] -= (cfg.repro_transfer + cfg.landauer_cost)
        nc = idx.size
        childpos = (self.pos[idx] + cfg.child_scatter * self.rng.standard_normal((nc, 2))) % cfg.W
        self.G = np.concatenate([self.G, children], 0)
        self.h = np.concatenate([self.h, np.zeros((nc, cfg.hidden))], 0)
        self.pos = np.concatenate([self.pos, childpos], 0)
        self.Ns = np.concatenate([self.Ns, np.full(nc, cfg.repro_transfer)])
        self.depth = np.concatenate([self.depth, np.full(nc, cfg.th.dV_max * 0.9)])
        self.generation = np.concatenate([self.generation, self.generation[idx] + 1])
        if self.size > cfg.pop_cap:
            keep = self.rng.permutation(self.size)[: cfg.pop_cap]
            mask = np.zeros(self.size, dtype=bool); mask[keep] = True
            self._compact(mask)

    def metrics(self) -> Dict[str, float]:
        chemo = float(np.mean(self._chemo[-2000:])) if self._chemo else float("nan")
        return dict(pop=self.size,
                    mean_gen=float(self.generation.mean()) if self.size else 0.0,
                    mean_Ns=float(self.Ns.mean()) if self.size else 0.0,
                    field_total=self.field.total(),
                    chemotaxis=chemo,
                    brain=float((self.h ** 2).mean()) if self.size else 0.0)


@dataclass
class WorldResult:
    history: List[dict]
    final_pop: int
    ended: str
    cfg: WorldConfig


def run(cfg: WorldConfig) -> WorldResult:
    w = World(cfg)
    history = []
    ended = "completed"
    for t in range(1, cfg.ticks + 1):
        w.step(t)
        if w.size == 0:
            ended = "extinct"; history.append(dict(t=t, pop=0)); break
        if t % cfg.probe_interval == 0:
            history.append(dict(t=t, **w.metrics()))
            w._chemo.clear()
    return WorldResult(history=history, final_pop=w.size, ended=ended, cfg=cfg)
