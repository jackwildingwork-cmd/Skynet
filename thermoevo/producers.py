"""The producer (autotroph) trophic level: processes that couple to a UNIVERSAL
gradient (sunlight) and fix it into standing structural stock (vegetation).

This is the environment getting bigger. Previously the "gradient" the herbivores
foraged was an abstract logistic field. Now that field is produced: it is the
standing biomass of a population of autotroph processes, each a first-class
Persistence-Theorem coupler (N_s structural stock under Langevin first-passage
dynamics, Omega Kramers well governing replication fidelity, forced-error seeding).

The gradient they couple to is *universal*, which is the whole point and the
sharp contrast with the herbivores:

  - Sunlight is available to every producer everywhere. There is nothing to FIND.
    A producer does not need chemotaxis or a world-model to reach its gradient.
  - Finiteness comes not from scarcity-of-location but from SELF-SHADING: producers
    sharing an area split the light between them (Beer-Lambert). A universal
    gradient still delivers only finite *accessible* flux per coupler. That is what
    caps producer biomass and makes producer-vs-producer competition real.

So producer selection acts on allocation (grow-fast-and-flimsy vs slow-and-standing,
and how far to disperse seed) — NOT on foraging intelligence, because there is no
foraging problem. Only the herbivores, whose gradient (producer biomass) is patchy
and must be found, are pushed toward intelligence. That difference falls straight
out of "universal vs patchy gradient" and is measured, not designed.

The producers' standing biomass, rasterised to a grid, is exactly the vegetation
field the herbivores sense and graze. This is RGC's "structural stock IS a gradient
for the next consumer", realised within a single tier: producers -> consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from .constants import Thermo


@dataclass
class ProducerConfig:
    th: Thermo = dc_field(default_factory=lambda: Thermo(dE_copy_max=4.0))
    W: int = 48
    init_pop: int = 200
    # universal sunlight: a flux available at EVERY cell to EVERY producer. Not
    # patchy, not found. (A global seasonal multiplier can scale it; base is flat.)
    light: float = 5.0
    shade_k: float = 8.0          # half-saturation biomass for self-shading share
    photo_eff: float = 1.0        # captured light -> structural stock
    respire: float = 0.07         # maintenance cost per unit standing biomass
    gamma_n: float = 0.02         # N_s leak (absorbing-boundary drift term)
    dt: float = 1.0
    homeostasis_spend: float = 0.5
    # seeding (reproduction): forced-error, paid from the parent's own stock
    seed_threshold: float = 24.0
    seed_transfer: float = 10.0
    landauer_cost: float = 1.5
    seed_scatter: float = 2.2     # base dispersal sigma (gene modulates)
    establish_cap: float = 14.0   # a seed establishes only on a "safe site": a cell
                                  # whose standing biomass is below this. Recruitment
                                  # limitation -> a spatial carrying capacity set by
                                  # safe-site availability, not the growth asymptote.
    mut_scale: float = 1.0
    pop_cap: int = 6000
    # grazing refuge: herbivores cannot strip more than this fraction of a cell's
    # standing biomass in a single tick, so ungrazed cells are a spatial refuge.
    graze_max_frac: float = 0.55
    # herbivore-facing sensing grid
    diffuse: float = 0.20
    sense_cap: float = 60.0       # nominal cap for normalising the herbivore's input
    seed: int = 0


GENE_DIM = 2   # [growth_allocation, dispersal] via sigmoids


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


class Producers:
    """A population of autotroph processes fixing universal sunlight into biomass."""

    def __init__(self, cfg: ProducerConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        n = cfg.init_pop
        self.pos = rng.random((n, 2)) * cfg.W
        self.Ns = np.full(n, cfg.seed_threshold * 0.6)
        self.depth = np.full(n, cfg.th.dV_max)
        self.gene = 0.4 * rng.standard_normal((n, GENE_DIM))
        self.generation = np.ones(n, dtype=int)
        self._grid = np.zeros((cfg.W, cfg.W))
        self.light_mult = 1.0     # external seasonal multiplier (1 = flat universal)

    @property
    def size(self) -> int:
        return self.pos.shape[0]

    def _cells(self, pos):
        W = self.cfg.W
        ix = np.floor(pos[:, 0]).astype(int) % W
        iy = np.floor(pos[:, 1]).astype(int) % W
        return ix, iy

    def _biomass_at_cells(self, ix, iy):
        """Standing biomass summed per cell, read back at each producer's cell."""
        W = self.cfg.W
        flat = ix * W + iy
        cell = np.zeros(W * W)
        np.add.at(cell, flat, self.Ns)
        return cell, flat

    def step(self) -> None:
        cfg = self.cfg
        if self.size == 0:
            return
        ix, iy = self._cells(self.pos)
        cell_biomass, flat = self._biomass_at_cells(ix, iy)
        local_B = cell_biomass[flat]                       # biomass in my cell (incl. me)

        # --- couple to the UNIVERSAL gradient, split by self-shading -------------
        # Every producer sees the same light everywhere; its captured share is
        # b_i / (shade_k + B_local) of it (Beer-Lambert). Lone producers capture
        # nearly all the local light; crowded ones split it. Finite flux from an
        # infinite/universal gradient -> a real carrying capacity, no "finding".
        L = cfg.light * self.light_mult
        share = self.Ns / (cfg.shade_k + local_B)
        captured = L * share
        gain = cfg.photo_eff * captured

        respire = cfg.respire * self.Ns
        F = gain - respire
        noise = np.sqrt(2.0 * cfg.gamma_n * cfg.th.kBT * cfg.dt) * self.rng.standard_normal(self.size)
        self.Ns = self.Ns + (F - cfg.gamma_n * self.Ns) * cfg.dt + noise

        # --- CII: Omega well maintenance + Kramers structural death -------------
        damage = cfg.th.damage_rate * cfg.th.dV_max * cfg.dt
        repair = cfg.th.repair_gain * cfg.homeostasis_spend * cfg.dt
        self.depth = np.clip(self.depth - damage + repair, 0.0, cfg.th.dV_max)
        p_escape = 1.0 - np.exp(-cfg.th.omega0 * np.exp(-self.depth / cfg.th.kBT) * cfg.dt)
        escaped = self.rng.random(self.size) < p_escape

        alive = (self.Ns > 0.0) & (~escaped)
        self._compact(alive)
        if self.size == 0:
            return

        # --- reproduction: forced-error seeding, paid from own stock ------------
        can = (self.Ns > cfg.seed_threshold) & (self.Ns > cfg.seed_transfer + cfg.landauer_cost)
        idx = np.nonzero(can)[0]
        if idx.size:
            self._seed(idx)

    def _compact(self, alive):
        self.pos = self.pos[alive]; self.Ns = self.Ns[alive]
        self.depth = self.depth[alive]; self.gene = self.gene[alive]
        self.generation = self.generation[alive]

    def _seed(self, idx):
        cfg = self.cfg
        omega = self.depth[idx] / cfg.th.dV_max
        mu = np.exp(-omega * cfg.th.dE_copy_max / cfg.th.kBT)          # forced error rate
        g = self.gene[idx]
        children = g + (mu[:, None] * cfg.mut_scale) * self.rng.standard_normal(g.shape)
        # parent pays the full seeding cost whether or not the seed finds a safe site
        self.Ns[idx] -= (cfg.seed_transfer + cfg.landauer_cost)
        nc = idx.size
        disp = cfg.seed_scatter * (0.4 + 1.2 * _sig(g[:, 1]))          # dispersal gene
        childpos = (self.pos[idx] + disp[:, None] * self.rng.standard_normal((nc, 2))) % cfg.W
        # recruitment limitation: a seed establishes only on a safe site (a cell whose
        # standing biomass is below establish_cap). This is the spatial density
        # regulator, decoupled from the growth energetics -> no knife-edge.
        cix, ciy = self._cells(childpos)
        W = cfg.W
        cell_now = np.zeros(W * W)
        pix, piy = self._cells(self.pos)
        np.add.at(cell_now, pix * W + piy, self.Ns)
        safe = cell_now[cix * W + ciy] < cfg.establish_cap
        if not safe.any():
            return
        idx, children, childpos = idx[safe], children[safe], childpos[safe]
        nc = idx.size
        self.pos = np.concatenate([self.pos, childpos], 0)
        self.Ns = np.concatenate([self.Ns, np.full(nc, cfg.seed_transfer)])
        self.depth = np.concatenate([self.depth, np.full(nc, cfg.th.dV_max * 0.9)])
        self.gene = np.concatenate([self.gene, children], 0)
        self.generation = np.concatenate([self.generation, self.generation[idx] + 1])
        if self.size > cfg.pop_cap:
            keep = self.rng.permutation(self.size)[: cfg.pop_cap]
            mask = np.zeros(self.size, dtype=bool); mask[keep] = True
            self._compact(mask)

    # --- herbivore-facing interface: the vegetation field -----------------------

    def biomass_grid(self) -> np.ndarray:
        """Rasterise standing biomass to a grid and diffuse a little so herbivores
        get a climbable gradient toward producer clumps. THIS is the vegetation the
        herbivores sense and forage."""
        cfg = self.cfg
        W = cfg.W
        g = np.zeros(W * W)
        if self.size:
            ix, iy = self._cells(self.pos)
            np.add.at(g, ix * W + iy, self.Ns)
        g = g.reshape(W, W)
        if cfg.diffuse > 0:
            lap = (np.roll(g, 1, 0) + np.roll(g, -1, 0) +
                   np.roll(g, 1, 1) + np.roll(g, -1, 1) - 4.0 * g)
            g = g + cfg.diffuse * lap
        self._grid = g
        return g

    def graze(self, hix: np.ndarray, hiy: np.ndarray, demand: np.ndarray) -> np.ndarray:
        """Herbivores at cells (hix,hiy) each want `demand` biomass. They consume the
        standing biomass of producers sharing their cell (scramble among herbivores;
        removal split across producers in the cell in proportion to biomass). A cell
        can lose at most graze_max_frac of its biomass per tick (refuge). Returns
        per-herbivore intake and depletes the producers' structural stock.
        """
        cfg = self.cfg
        W = cfg.W
        nH = hix.shape[0]
        got = np.zeros(nH)
        if self.size == 0 or nH == 0:
            return got
        pix, piy = self._cells(self.pos)
        pflat = pix * W + piy
        prod_cell = np.zeros(W * W)
        np.add.at(prod_cell, pflat, self.Ns)

        hflat = hix * W + hiy
        hd = np.zeros(W * W)
        np.add.at(hd, hflat, demand)

        avail_cell = cfg.graze_max_frac * prod_cell            # refuge cap
        d_here = hd[hflat]
        avail_here = avail_cell[hflat]
        share = np.where(d_here > 0, np.minimum(1.0, avail_here / d_here), 0.0)
        got = demand * share

        # total removed per cell, then deplete producers in that cell proportionally
        taken_cell = np.zeros(W * W)
        np.add.at(taken_cell, hflat, got)
        pc = prod_cell[pflat]
        frac = np.where(pc > 0, taken_cell[pflat] / pc, 0.0)
        self.Ns = np.maximum(0.0, self.Ns - frac * self.Ns)
        return got

    def metrics(self) -> dict:
        if self.size == 0:
            return dict(prod_pop=0, prod_biomass=0.0, prod_gen=0.0,
                        prod_alloc=float("nan"), prod_disp=float("nan"))
        return dict(prod_pop=self.size,
                    prod_biomass=float(self.Ns.sum()),
                    prod_gen=float(self.generation.mean()),
                    prod_alloc=float(_sig(self.gene[:, 0]).mean()),
                    prod_disp=float(_sig(self.gene[:, 1]).mean()))
