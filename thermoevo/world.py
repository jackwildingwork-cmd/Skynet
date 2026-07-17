"""Spatial evolutionary world, now with TWO trophic levels in one tier.

Level 1 — producers (autotrophs, `producers.py`): processes that couple to a
UNIVERSAL gradient (sunlight, available to every producer everywhere) and fix it
into standing structural stock (vegetation). Finiteness comes from self-shading,
not from having to find the gradient.

Level 2 — herbivores (this module): the foragers. They CANNOT access sunlight.
Their gradient is the producers' standing biomass, which is patchy and mobile and
must be FOUND. Each tick a herbivore senses local vegetation (producer biomass
rasterised to a grid) and its own hunger, its evolved recurrent controller outputs
a move, it pays a search cost, and it grazes producer biomass at its new cell
(competing with any herbivores sharing it, limited by a refuge cap). Condition I is
explicit and local: intake must exceed the cost of finding and consuming, or N_s
drifts to the absorbing boundary and the herbivore dies.

The sharp contrast between the levels is the point: only the herbivores face a
find-it problem, so only they are pushed toward foraging intelligence. Producers,
facing a universal gradient, are selected on allocation instead. Both are the same
validated thermodynamic core (N_s Langevin + absorbing boundary, Omega Kramers
wells, forced-error reproduction); the environment, not the code, makes them differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, List

import numpy as np

from .constants import Thermo
from .producers import Producers, ProducerConfig
from .neuralnet import NetSpec, mutate


@dataclass
class WorldConfig:
    th: Thermo = dc_field(default_factory=lambda: Thermo(dE_copy_max=4.0))
    W: int = 48
    # producer (autotroph) layer — the vegetation is now grown, not given
    producers: ProducerConfig = dc_field(default_factory=ProducerConfig)
    # herbivore / controller
    hidden: int = 5
    init_pop: int = 60
    init_scale: float = 0.5
    pop_cap: int = 4000              # generous: herbivore numbers are set by the
                                     # producers' productivity (food), not by this cap
    max_step: float = 1.5
    max_intake: float = 8.0
    # energetics (Condition I: intake must beat the cost of finding + consuming).
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
    # n-1 coordination (multicellularity): adhesive same-clade cells that share a
    # location bond into an organism and POOL structural stock N_s (rich cells
    # subsidise starving kin), buffering members against the boom-bust boundary.
    # Off by default so the validated foraging core is unchanged.
    adhesion: bool = False
    clade_bins: int = 24             # kin resolution: same bin = same clade
    clade_drift: float = 0.01        # heritable clade tag drift per birth
    bond_thr: float = 0.5            # sigmoid(adhesion gene) above this -> will bond
    share: float = 0.25             # fraction of the gap to the organism mean, per tick
    share_overhead: float = 0.03    # N_s cost of being coordinated, per bonded cell/tick
    cohesion: float = 0.0            # aggregation: pull toward same-clade neighbours,
                                     # scaled by sigmoid(adhesion). 0 = no aggregation
                                     # (adhesive kin must meet by chance). This is the
                                     # prerequisite the sharing benefit needs to pay.
    # group-level reproduction (fitness export): an organism that is large and rich
    # enough BUDS a propagule as a unit. This is what makes cooperation pay — it
    # exports fitness to the higher level (Michod). Off by default.
    group_repro: bool = False
    org_repro_min: int = 4           # organism must have >= this many cells to bud
    org_repro_thr: float = 60.0      # ... and this much pooled N_s
    org_propagule: int = 3           # cells released per budding event
    org_propagule_cost: float = 26.0 # N_s levied across the organism to bud
    # third trophic level: predators that hunt herbivores. Their gradient is the
    # herbivore population itself — mobile prey that must be chased and caught (a
    # harder foraging problem than grazing). init_predators=0 -> no predators, the
    # two-level world is unchanged.
    init_predators: int = 0
    pred_hidden: int = 5
    pred_init_scale: float = 0.5
    pred_max_step: float = 1.9       # faster than herbivores (max_step 1.5): chase
    pred_max_intake: float = 25.0
    pred_occupancy: float = 0.3
    pred_move_cost: float = 0.4
    pred_homeostasis: float = 0.4
    pred_gamma_n: float = 0.05
    pred_repro_threshold: float = 32.0
    pred_repro_transfer: float = 14.0
    pred_landauer_cost: float = 2.0
    pred_child_scatter: float = 2.0
    pred_pop_cap: int = 1500
    catch_frac: float = 0.50         # max fraction of a cell's prey biomass caught/tick
    catch_efficiency: float = 0.9    # killed prey biomass -> predator structural stock
    prey_diffuse: float = 0.20       # spread herbivore biomass into a climbable field
    prey_sense_cap: float = 40.0
    # run
    ticks: int = 4000
    probe_interval: int = 200
    seed: int = 0

    def __post_init__(self):
        # keep the producer layer on the same grid and RNG-stream seed
        if self.producers.W != self.W:
            self.producers.W = self.W


IN_DIM, OUT_DIM = 4, 2   # sense [veg_here, grad_x, grad_y, hunger] -> move [vx, vy]


class World:
    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.net = NetSpec(IN_DIM, cfg.hidden, OUT_DIM)
        self.producers = Producers(cfg.producers, self.rng)
        # let producers settle to a standing crop before herbivores arrive, so the
        # foragers have a real (grown) field to find rather than an empty world.
        for _ in range(60):
            self.producers.step()
        n = cfg.init_pop
        self.G = self.net.random(n, cfg.init_scale, self.rng)
        self.h = np.zeros((n, cfg.hidden))
        # spawn herbivore founders ON producers (where the food is), jittered.
        if self.producers.size:
            pick = self.rng.integers(0, self.producers.size, size=n)
            base = self.producers.pos[pick]
        else:
            base = self.rng.random((n, 2)) * cfg.W
        self.pos = (base + self.rng.standard_normal((n, 2))) % cfg.W
        self.Ns = np.full(n, cfg.repro_threshold * 0.6)
        self.depth = np.full(n, cfg.th.dV_max)
        self.generation = np.ones(n, dtype=int)
        # n-1 coordination state (carried always; only used when cfg.adhesion)
        self.adh = self.rng.standard_normal(n) * 0.5          # adhesion gene (logit)
        self.clade = self.rng.random(n)                       # heritable kin tag in [0,1)
        self.bonded = np.zeros(n, dtype=bool)                 # in an organism this tick
        self._chemo = []   # rolling chemotaxis alignment samples
        self._last_org = np.array([], dtype=int)              # organism sizes this tick
        # cumulative buffering accounting (starvation deaths, bonded vs solitary)
        self._b_alive = 0; self._b_starved = 0; self._s_alive = 0; self._s_starved = 0
        # --- third trophic level: predators hunting herbivores ------------------
        self._pred_chemo = []
        self._pred_kills = 0
        m = cfg.init_predators
        if m > 0:
            self.pnet = NetSpec(IN_DIM, cfg.pred_hidden, OUT_DIM)
            self.Gp = self.pnet.random(m, cfg.pred_init_scale, self.rng)
            self.hp = np.zeros((m, cfg.pred_hidden))
            # start predators where the prey are (on the herbivore founders)
            pick = self.rng.integers(0, self.size, size=m)
            self.pos_p = (self.pos[pick] + self.rng.standard_normal((m, 2))) % cfg.W
            self.Ns_p = np.full(m, cfg.pred_repro_threshold * 0.6)
            self.depth_p = np.full(m, cfg.th.dV_max)
            self.gen_p = np.ones(m, dtype=int)
        else:
            self.pnet = None
            self.Gp = np.zeros((0, 0)); self.hp = np.zeros((0, cfg.pred_hidden))
            self.pos_p = np.zeros((0, 2)); self.Ns_p = np.zeros(0)
            self.depth_p = np.zeros(0); self.gen_p = np.zeros(0, dtype=int)

    @property
    def size(self) -> int:
        return self.G.shape[0]

    def _cells(self, pos):
        W = self.cfg.W
        ix = np.floor(pos[:, 0]).astype(int) % W
        iy = np.floor(pos[:, 1]).astype(int) % W
        return ix, iy

    def _sense(self, grid, ix, iy):
        W = self.cfg.W
        here = grid[ix, iy]
        gx = grid[(ix + 1) % W, iy] - grid[(ix - 1) % W, iy]
        gy = grid[ix, (iy + 1) % W] - grid[ix, (iy - 1) % W]
        return here, gx, gy

    def step(self, t: int) -> None:
        cfg = self.cfg
        # --- level 1: producers fix universal sunlight into vegetation ----------
        self.producers.step()
        grid = self.producers.biomass_grid()
        if self.size == 0:
            return

        # --- level 2: herbivores forage the produced vegetation -----------------
        ix, iy = self._cells(self.pos)
        here, gx, gy = self._sense(grid, ix, iy)
        cap = cfg.producers.sense_cap
        hunger = np.clip(1.0 - self.Ns / cfg.repro_threshold, 0.0, 1.0)
        x = np.stack([here / cap, gx, gy, hunger], axis=1)

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
        # aggregation: adhesive cells drift toward same-clade neighbours (the
        # prerequisite for bonding). Driven by the same adhesion gene.
        if cfg.adhesion and cfg.cohesion > 0.0:
            self._cohere()
        nix, niy = self._cells(self.pos)
        intake = self.producers.graze(nix, niy, np.full(self.size, cfg.max_intake))

        move_cost = cfg.move_cost * step_len
        F_drift = intake - cfg.occupancy - move_cost - cfg.homeostasis_spend
        noise = np.sqrt(2.0 * cfg.gamma_n * cfg.th.kBT * cfg.dt) * self.rng.standard_normal(self.size)
        self.Ns = self.Ns + (F_drift - cfg.gamma_n * self.Ns) * cfg.dt + noise

        # n-1 coordination: bonded organisms pool N_s BEFORE the death boundary is
        # tested, so a starving cell can be rescued by richer clade-mates.
        bonded = np.zeros(self.size, dtype=bool)
        if cfg.adhesion:
            bonded, self._last_org, comps = self._organisms_share(nix, niy)
            if cfg.group_repro and comps:
                self._group_reproduce(comps)                    # organisms bud as units
                if self.size > bonded.size:
                    bonded = np.concatenate([bonded, np.zeros(self.size - bonded.size, dtype=bool)])
            self.bonded = bonded

        # integrity maintenance + Kramers escape (structural death)
        damage = cfg.th.damage_rate * cfg.th.dV_max * cfg.dt
        repair = cfg.th.repair_gain * cfg.homeostasis_spend * cfg.dt
        self.depth = np.clip(self.depth - damage + repair, 0.0, cfg.th.dV_max)
        p_escape = 1.0 - np.exp(-cfg.th.omega0 * np.exp(-self.depth / cfg.th.kBT) * cfg.dt)
        escaped = self.rng.random(self.size) < p_escape

        # --- level 3: predators hunt the herbivores (a mobile, fleeing gradient) --
        predated = np.zeros(self.size, dtype=bool)
        if self.pnet is not None and self.Ns_p.size:
            predated = self._predators_step(nix, niy)

        alive = (self.Ns > 0.0) & (~escaped) & (~predated)
        if cfg.adhesion:
            # attribute starvation (N_s boundary) deaths to bonded vs solitary cells
            starved = self.Ns <= 0.0
            self._b_alive += int(bonded.sum()); self._b_starved += int((bonded & starved).sum())
            self._s_alive += int((~bonded).sum()); self._s_starved += int((~bonded & starved).sum())
        self._compact(alive)
        if self.size == 0:
            return
        can = (self.Ns > cfg.repro_threshold) & (self.Ns > cfg.repro_transfer + cfg.landauer_cost)
        if cfg.adhesion and cfg.group_repro:
            can = can & (~self.bonded)               # bonded cells breed only via the organism
        idx = np.nonzero(can)[0]
        if idx.size:
            self._reproduce(idx)

    @property
    def pred_size(self) -> int:
        return self.Ns_p.shape[0]

    def _predators_step(self, hix, hiy):
        """The third trophic level. Predators sense a herbivore-density field (prey
        biomass rasterised + diffused into a climbable gradient), their evolved
        controller chooses a move (they are faster than herbivores — a chase), and
        they catch prey in their cell: up to catch_frac of the cell's herbivore
        biomass, limited by predator demand, killing those herbivores and converting
        the kill into predator structural stock. Same core (N_s, Omega, forced-error
        reproduction). Returns the boolean mask of herbivores killed this tick."""
        cfg = self.cfg; W = cfg.W; N2 = W * W
        hflat = hix * W + hiy
        praw = np.zeros(N2); np.add.at(praw, hflat, self.Ns)          # prey biomass per cell
        prey = praw.reshape(W, W)
        if cfg.prey_diffuse > 0:
            lap = (np.roll(prey, 1, 0) + np.roll(prey, -1, 0) +
                   np.roll(prey, 1, 1) + np.roll(prey, -1, 1) - 4.0 * prey)
            prey = prey + cfg.prey_diffuse * lap
        pix, piy = self._cells(self.pos_p)
        here, gx, gy = self._sense(prey, pix, piy)
        hunger = np.clip(1.0 - self.Ns_p / cfg.pred_repro_threshold, 0.0, 1.0)
        xp = np.stack([here / cfg.prey_sense_cap, gx, gy, hunger], axis=1)
        outp, self.hp = self.pnet.step(self.Gp, self.hp, xp)
        movep = cfg.pred_max_step * np.tanh(outp)
        slen = np.linalg.norm(movep, axis=1)
        grad = np.stack([gx, gy], axis=1); gn = np.linalg.norm(grad, axis=1)
        good = (gn > 1e-6) & (slen > 1e-6)
        if good.any():                                                # hunting intelligence
            self._pred_chemo.append(float(((movep[good] * grad[good]).sum(1) /
                                           (slen[good] * gn[good])).mean()))
        self.pos_p = (self.pos_p + movep) % W
        ppix, ppiy = self._cells(self.pos_p)
        pflat = ppix * W + ppiy
        m = self.pred_size
        demand = np.full(m, cfg.pred_max_intake)
        hcell = np.zeros(N2); np.add.at(hcell, hflat, self.Ns)
        pdem = np.zeros(N2); np.add.at(pdem, pflat, demand)
        caught_cell = np.minimum(cfg.catch_frac * hcell, pdem)        # biomass taken per cell
        pkill = np.where(hcell > 0.0, caught_cell / np.maximum(hcell, 1e-9), 0.0)
        predated = self.rng.random(self.size) < pkill[hflat]          # which herbivores die
        killed_cell = np.zeros(N2); np.add.at(killed_cell, hflat[predated], self.Ns[predated])
        share = np.where(pdem[pflat] > 0.0, demand / pdem[pflat], 0.0)
        intake_p = killed_cell[pflat] * share * cfg.catch_efficiency
        self._pred_kills += int(predated.sum())
        # predator energetics + Kramers death + reproduction
        move_cost = cfg.pred_move_cost * slen
        F = intake_p - cfg.pred_occupancy - move_cost - cfg.pred_homeostasis
        noise = np.sqrt(2.0 * cfg.pred_gamma_n * cfg.th.kBT * cfg.dt) * self.rng.standard_normal(m)
        self.Ns_p = self.Ns_p + (F - cfg.pred_gamma_n * self.Ns_p) * cfg.dt + noise
        damage = cfg.th.damage_rate * cfg.th.dV_max * cfg.dt
        repair = cfg.th.repair_gain * cfg.pred_homeostasis * cfg.dt
        self.depth_p = np.clip(self.depth_p - damage + repair, 0.0, cfg.th.dV_max)
        p_esc = 1.0 - np.exp(-cfg.th.omega0 * np.exp(-self.depth_p / cfg.th.kBT) * cfg.dt)
        palive = (self.Ns_p > 0.0) & (self.rng.random(m) >= p_esc)
        self._compact_pred(palive)
        if self.pred_size:
            pcan = ((self.Ns_p > cfg.pred_repro_threshold) &
                    (self.Ns_p > cfg.pred_repro_transfer + cfg.pred_landauer_cost))
            pidx = np.nonzero(pcan)[0]
            if pidx.size:
                self._reproduce_pred(pidx)
        return predated

    def _compact_pred(self, alive):
        self.Gp = self.Gp[alive]; self.hp = self.hp[alive]; self.pos_p = self.pos_p[alive]
        self.Ns_p = self.Ns_p[alive]; self.depth_p = self.depth_p[alive]
        self.gen_p = self.gen_p[alive]

    def _reproduce_pred(self, idx):
        cfg = self.cfg
        omega = self.depth_p[idx] / cfg.th.dV_max
        mu = np.exp(-omega * cfg.th.dE_copy_max / cfg.th.kBT)
        children = mutate(self.Gp[idx], mu, cfg.mut_scale, self.rng)
        self.Ns_p[idx] -= (cfg.pred_repro_transfer + cfg.pred_landauer_cost)
        nc = idx.size
        childpos = (self.pos_p[idx] + cfg.pred_child_scatter * self.rng.standard_normal((nc, 2))) % cfg.W
        self.Gp = np.concatenate([self.Gp, children], 0)
        self.hp = np.concatenate([self.hp, np.zeros((nc, cfg.pred_hidden))], 0)
        self.pos_p = np.concatenate([self.pos_p, childpos], 0)
        self.Ns_p = np.concatenate([self.Ns_p, np.full(nc, cfg.pred_repro_transfer)])
        self.depth_p = np.concatenate([self.depth_p, np.full(nc, cfg.th.dV_max * 0.9)])
        self.gen_p = np.concatenate([self.gen_p, self.gen_p[idx] + 1])
        if self.pred_size > cfg.pred_pop_cap:
            keep = self.rng.permutation(self.pred_size)[: cfg.pred_pop_cap]
            mask = np.zeros(self.pred_size, dtype=bool); mask[keep] = True
            self._compact_pred(mask)

    def _cohere(self):
        """Pull each cell toward the centroid of same-clade cells in its 3x3
        neighbourhood, with strength cohesion * sigmoid(adhesion). Adhesive kin
        aggregate; non-adhesive cells barely move. Vectorised over a dense
        (cell x clade-bin) grid."""
        cfg = self.cfg; W = cfg.W; B = cfg.clade_bins
        ix, iy = self._cells(self.pos)
        cbin = np.floor(self.clade * B).astype(int) % B
        key = (ix * W + iy) * B + cbin
        K = W * W * B
        gc = np.zeros(K); gx = np.zeros(K); gy = np.zeros(K)
        np.add.at(gc, key, 1.0)
        np.add.at(gx, key, self.pos[:, 0]); np.add.at(gy, key, self.pos[:, 1])
        sx = np.zeros(self.size); sy = np.zeros(self.size); sc = np.zeros(self.size)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nkey = (((ix + dx) % W) * W + ((iy + dy) % W)) * B + cbin
                sx += gx[nkey]; sy += gy[nkey]; sc += gc[nkey]
        adh = 1.0 / (1.0 + np.exp(-self.adh))
        k = (cfg.cohesion * adh)[:, None]
        target = np.stack([sx / sc, sy / sc], axis=1)
        self.pos = (self.pos + k * (target - self.pos)) % W

    def _organisms_share(self, ix, iy):
        """Bond adhesive (sigmoid(adh) >= bond_thr) same-clade cells that lie in the
        same OR an adjacent cell into an organism (a connected neighbourhood, so
        members occupy DIFFERENT cells and graze without scrambling against each
        other), and pool their N_s toward the organism mean (minus a coordination
        overhead). Pooling conserves N_s within the organism, so it is pure variance
        reduction across kin — group-level homeostasis (CII). Returns a per-cell
        `bonded` mask and the organism sizes. O(N) via cell/clade representatives."""
        cfg = self.cfg; B = cfg.clade_bins
        adh = 1.0 / (1.0 + np.exp(-self.adh))
        idx_e = np.nonzero(adh >= cfg.bond_thr)[0]
        bonded = np.zeros(self.size, dtype=bool)
        if idx_e.size < 2:
            return bonded, np.array([], dtype=int), []
        exx = ix[idx_e]; eyy = iy[idx_e]
        cbin = (np.floor(self.clade[idx_e] * B).astype(int) % B)
        ne = idx_e.size
        parent = list(range(ne))
        def find(x):
            r = x
            while parent[r] != r: r = parent[r]
            while parent[x] != r: parent[x], x = r, parent[x]
            return r
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[ra] = rb
        # representative local-index per (cell, clade-bin); union same-key members
        reps = {}
        for a in range(ne):
            k = (exx[a], eyy[a], cbin[a])
            r = reps.get(k)
            if r is None: reps[k] = a
            else: union(a, r)
        # union representatives across adjacent cells sharing a clade-bin
        for (cx, cy, cb), a in reps.items():
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                nb = reps.get((cx + dx, cy + dy, cb))
                if nb is not None: union(a, nb)
        # gather components
        groups = {}
        for a in range(ne):
            r = find(a); groups.setdefault(r, []).append(a)
        sizes = []; comps = []
        for mem in groups.values():
            if len(mem) < 2: continue
            gi = idx_e[mem]
            mean = self.Ns[gi].mean()
            self.Ns[gi] += cfg.share * (mean - self.Ns[gi]) - cfg.share_overhead
            bonded[gi] = True; sizes.append(len(mem)); comps.append(gi)
        return bonded, np.array(sizes, dtype=int), comps

    def _group_reproduce(self, comps):
        """The organism reproduces as a higher-level individual (fitness export): each
        member rich enough to breed buds ONE offspring, at the same per-cell cost as
        solitary reproduction, but the offspring are released together at the
        organism's centroid so they re-form an organism (assortment). Bonded members
        do NOT also reproduce individually (see step) — their fitness is the group's.
        The only differences from solitary reproduction are therefore (a) prior N_s
        sharing and (b) offspring assortment; whether that nets positive is left to
        selection."""
        cfg = self.cfg
        gsz, gpos, gadh, gcl, ggen, gdep = [], [], [], [], [], []
        for gi in comps:
            if gi.size < cfg.org_repro_min:
                continue
            rich = gi[(self.Ns[gi] > cfg.repro_threshold) &
                      (self.Ns[gi] > cfg.repro_transfer + cfg.landauer_cost)]
            if rich.size == 0:
                continue
            centroid = self.pos[gi].mean(0)
            omega = self.depth[rich] / cfg.th.dV_max
            mu = np.exp(-omega * cfg.th.dE_copy_max / cfg.th.kBT)
            kids = mutate(self.G[rich], mu, cfg.mut_scale, self.rng)
            self.Ns[rich] -= (cfg.repro_transfer + cfg.landauer_cost)
            for j in range(rich.size):
                gsz.append(kids[j])
                gpos.append((centroid + cfg.child_scatter * self.rng.standard_normal(2)) % cfg.W)
                gadh.append(self.adh[rich[j]] + cfg.mut_scale * mu[j] * self.rng.standard_normal())
                gcl.append((self.clade[rich[j]] + cfg.clade_drift * self.rng.standard_normal()) % 1.0)
                ggen.append(int(self.generation[rich[j]]) + 1)
                gdep.append(cfg.th.dV_max * 0.9)
        if not gsz:
            return
        nc = len(gsz)
        self.G = np.concatenate([self.G, np.stack(gsz)], 0)
        self.h = np.concatenate([self.h, np.zeros((nc, cfg.hidden))], 0)
        self.pos = np.concatenate([self.pos, np.stack(gpos)], 0)
        self.Ns = np.concatenate([self.Ns, np.full(nc, cfg.repro_transfer)])
        self.depth = np.concatenate([self.depth, np.array(gdep)])
        self.generation = np.concatenate([self.generation, np.array(ggen, dtype=int)])
        self.adh = np.concatenate([self.adh, np.array(gadh)])
        self.clade = np.concatenate([self.clade, np.array(gcl)])

    def _compact(self, alive):
        self.G = self.G[alive]; self.h = self.h[alive]; self.pos = self.pos[alive]
        self.Ns = self.Ns[alive]; self.depth = self.depth[alive]
        self.generation = self.generation[alive]
        self.adh = self.adh[alive]; self.clade = self.clade[alive]
        self.bonded = self.bonded[alive]

    def _reproduce(self, idx):
        cfg = self.cfg
        omega = self.depth[idx] / cfg.th.dV_max
        mu = np.exp(-omega * cfg.th.dE_copy_max / cfg.th.kBT)       # forced error rate
        children = mutate(self.G[idx], mu, cfg.mut_scale, self.rng)
        self.Ns[idx] -= (cfg.repro_transfer + cfg.landauer_cost)
        nc = idx.size
        childpos = (self.pos[idx] + cfg.child_scatter * self.rng.standard_normal((nc, 2))) % cfg.W
        # adhesion inherited with the same thermally-forced error; clade tag drifts
        child_adh = self.adh[idx] + cfg.mut_scale * mu * self.rng.standard_normal(nc)
        child_clade = (self.clade[idx] + cfg.clade_drift * self.rng.standard_normal(nc)) % 1.0
        self.G = np.concatenate([self.G, children], 0)
        self.h = np.concatenate([self.h, np.zeros((nc, cfg.hidden))], 0)
        self.pos = np.concatenate([self.pos, childpos], 0)
        self.Ns = np.concatenate([self.Ns, np.full(nc, cfg.repro_transfer)])
        self.depth = np.concatenate([self.depth, np.full(nc, cfg.th.dV_max * 0.9)])
        self.generation = np.concatenate([self.generation, self.generation[idx] + 1])
        self.adh = np.concatenate([self.adh, child_adh])
        self.clade = np.concatenate([self.clade, child_clade])
        self.bonded = np.concatenate([self.bonded, np.zeros(nc, dtype=bool)])
        if self.size > cfg.pop_cap:
            keep = self.rng.permutation(self.size)[: cfg.pop_cap]
            mask = np.zeros(self.size, dtype=bool); mask[keep] = True
            self._compact(mask)

    def metrics(self) -> Dict[str, float]:
        chemo = float(np.mean(self._chemo[-2000:])) if self._chemo else float("nan")
        m = dict(pop=self.size,
                 mean_gen=float(self.generation.mean()) if self.size else 0.0,
                 mean_Ns=float(self.Ns.mean()) if self.size else 0.0,
                 field_total=self.producers.metrics()["prod_biomass"],
                 chemotaxis=chemo,
                 brain=float((self.h ** 2).mean()) if self.size else 0.0)
        if self.size:
            m["mean_adh"] = float((1.0 / (1.0 + np.exp(-self.adh))).mean())
            m["max_org"] = int(self._last_org.max()) if self._last_org.size else 1
            m["mean_org"] = float(self._last_org.mean()) if self._last_org.size else 1.0
        if self.pnet is not None:
            pc = float(np.mean(self._pred_chemo[-2000:])) if self._pred_chemo else float("nan")
            m["pred_pop"] = self.pred_size
            m["pred_chemo"] = pc
            m["pred_gen"] = float(self.gen_p.mean()) if self.pred_size else 0.0
        m.update(self.producers.metrics())
        return m

    def buffering(self) -> Dict[str, float]:
        """Cumulative starvation-death rate for bonded vs solitary cells. If pooling
        buffers, bonded cells should starve at a LOWER rate than solitary ones."""
        b = self._b_starved / self._b_alive if self._b_alive else float("nan")
        s = self._s_starved / self._s_alive if self._s_alive else float("nan")
        return dict(bonded_death=b, solitary_death=s,
                    bonded_cell_ticks=self._b_alive, solitary_cell_ticks=self._s_alive)


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
            ended = "extinct"; history.append(dict(t=t, pop=0, **w.producers.metrics())); break
        if t % cfg.probe_interval == 0:
            history.append(dict(t=t, **w.metrics()))
            w._chemo.clear(); w._pred_chemo.clear()
    return WorldResult(history=history, final_pop=w.size, ended=ended, cfg=cfg)
