"""The evolutionary layer: a population of controller-genomes on a finite shared
gradient, with thermodynamically-forced variation and no imposed fitness.

Each tick, every living agent's controller sets a coupling action from the last
cue; agents compete for a FINITE gradient (proportional to match quality, so more
agents means less each — selection); captured free energy funds structural stock
N_s (Langevin, absorbing boundary) after paying basal, brain-upkeep, and
homeostatic costs; integrity Omega is held by homeostatic spend (Kramers wells);
agents die by resource exhaustion or Kramers escape; and when N_s clears a
threshold they reproduce by branching, copying the genome with forced error
mu = exp(-Omega*dE/kBT). Nothing selects except differential survival and
reproduction on the shared gradient.

Instrumentation ("what evolves") probes the living population periodically for
coupling skill, predictive information (does an internal model arise?), and brain
usage, so we can watch adaptation and the emergence — or not — of prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .constants import Thermo
from .environment import Gradient
from .genome import genome_size, random_genomes, batch_step, mutate, copy_error_rate
from .controller import predictive_information


@dataclass
class EvoConfig:
    th: Thermo = field(default_factory=Thermo)
    H: int = 3                      # controller hidden units
    init_pop: int = 40
    init_genome_scale: float = 0.1
    pop_cap: int = 600
    # environment
    rho: float = 0.9                # gradient predictability
    cue_noise: float = 0.5
    grad_width: float = 1.0
    G_total: float = 60.0           # finite shared gradient power per tick
    uniform_capture: bool = False   # True = no coupling-based selection (drift control)
    # energetics
    occupancy: float = 0.30
    homeostasis_spend: float = 0.30 # E_h per tick (maintains the Kramers well)
    brain_cost: float = 0.50        # upkeep per unit of internal-state activity <h^2>
    gamma_n: float = 0.05
    dt: float = 1.0
    # reproduction
    repro_threshold: float = 6.0
    repro_transfer: float = 2.5
    landauer_cost: float = 0.5      # ~ Ns_info * kBT ln2 (fixed proxy)
    mut_scale: float = 0.8          # weight-perturbation scale per unit error rate
    child_well_frac: float = 0.9    # newborn integrity as fraction of dV_max
    # run
    ticks: int = 4000
    probe_interval: int = 200
    probe_agents: int = 40
    probe_steps: int = 3000
    seed: int = 0


class Population:
    def __init__(self, cfg: EvoConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        n, H = cfg.init_pop, cfg.H
        self.G = random_genomes(n, H, cfg.init_genome_scale, self.rng)
        self.h = np.zeros((n, H))
        self.Ns = np.full(n, cfg.repro_threshold * 0.6)
        self.depth = np.full(n, cfg.th.dV_max)
        self.age = np.zeros(n)
        self.generation = np.ones(n, dtype=int)
        self.grad = Gradient(cfg.rho, cfg.cue_noise, G0=1.0, width=cfg.grad_width)
        self.theta_prev_cue = 0.0
        # precompute a long theta/cue stream lazily
        self._theta, self._cue = self.grad.run(cfg.ticks + 2, self.rng)

    @property
    def size(self) -> int:
        return self.G.shape[0]

    def step(self, t: int) -> None:
        cfg = self.cfg
        if self.size == 0:
            return
        theta_t = self._theta[t]
        cue_prev = self._cue[t - 1] if t > 0 else 0.0
        # controller acts on the last cue, matching the (as-yet-unseen) theta_t
        a, self.h = batch_step(self.G, self.h, cue_prev, cfg.H)
        # finite-gradient competition: capture proportional to match quality
        d = np.exp(-((a - theta_t) ** 2) / (2.0 * cfg.grad_width**2))
        if cfg.uniform_capture:
            # drift control: energy is shared equally, so coupling quality is NOT
            # selected; only forced variation + neutral demographic drift act.
            captured = np.full(self.size, cfg.G_total / self.size)
        else:
            dsum = d.sum()
            captured = cfg.G_total * d / dsum if dsum > 0 else np.zeros_like(d)
        brain = cfg.brain_cost * (self.h ** 2).mean(axis=1)
        F_drift = captured - cfg.occupancy - cfg.homeostasis_spend - brain
        # N_s Langevin (additive noise, absorbing boundary)
        noise = np.sqrt(2.0 * cfg.gamma_n * cfg.th.kBT * cfg.dt) * self.rng.standard_normal(self.size)
        self.Ns = self.Ns + (F_drift - cfg.gamma_n * self.Ns) * cfg.dt + noise
        # Kramers well maintenance (homeostasis)
        damage = cfg.th.damage_rate * cfg.th.dV_max * cfg.dt
        repair = cfg.th.repair_gain * cfg.homeostasis_spend * cfg.dt
        self.depth = np.clip(self.depth - damage + repair, 0.0, cfg.th.dV_max)
        self.age += cfg.dt

        # deaths: resource (Ns<=0) or Kramers escape (structural)
        escape_rate = cfg.th.omega0 * np.exp(-self.depth / cfg.th.kBT)
        p_escape = 1.0 - np.exp(-escape_rate * cfg.dt)
        escaped = self.rng.random(self.size) < p_escape
        alive = (self.Ns > 0.0) & (~escaped)
        self._compact(alive)

        # reproduction (branching): those above threshold fork
        if self.size == 0:
            return
        can = (self.Ns > cfg.repro_threshold) & (self.Ns > cfg.repro_transfer + cfg.landauer_cost)
        idx = np.nonzero(can)[0]
        if idx.size:
            self._reproduce(idx, t)

    def _compact(self, alive: np.ndarray) -> None:
        self.G = self.G[alive]; self.h = self.h[alive]; self.Ns = self.Ns[alive]
        self.depth = self.depth[alive]; self.age = self.age[alive]
        self.generation = self.generation[alive]

    def _reproduce(self, idx: np.ndarray, t: int) -> None:
        cfg = self.cfg
        omega = self.depth[idx] / cfg.th.dV_max
        mu = np.exp(-omega * cfg.th.dE_copy_max / cfg.th.kBT)   # forced error rate
        parents = self.G[idx]
        noise = self.rng.standard_normal(parents.shape)
        children = parents + cfg.mut_scale * mu[:, None] * noise
        # parent pays transfer + landauer floor
        self.Ns[idx] -= (cfg.repro_transfer + cfg.landauer_cost)
        n_child = idx.size
        childh = np.zeros((n_child, cfg.H))
        childNs = np.full(n_child, cfg.repro_transfer)
        childdepth = np.full(n_child, cfg.th.dV_max * cfg.child_well_frac)
        childage = np.zeros(n_child)
        childgen = self.generation[idx] + 1
        # append
        self.G = np.concatenate([self.G, children], axis=0)
        self.h = np.concatenate([self.h, childh], axis=0)
        self.Ns = np.concatenate([self.Ns, childNs])
        self.depth = np.concatenate([self.depth, childdepth])
        self.age = np.concatenate([self.age, childage])
        self.generation = np.concatenate([self.generation, childgen])
        if self.size > cfg.pop_cap:                    # hard safety cap
            keep = self.rng.permutation(self.size)[: cfg.pop_cap]
            mask = np.zeros(self.size, dtype=bool); mask[keep] = True
            self._compact(mask)

    def probe(self, rng: np.random.Generator) -> Dict[str, float]:
        """Run a sample of living genomes on a fresh gradient stream and measure
        coupling skill, predictive information, and brain usage."""
        cfg = self.cfg
        if self.size == 0:
            return dict(skill=float("nan"), pred_info=float("nan"), brain=float("nan"))
        m = min(cfg.probe_agents, self.size)
        sel = rng.choice(self.size, size=m, replace=False)
        G = self.G[sel]
        grad = Gradient(cfg.rho, cfg.cue_noise, G0=1.0, width=cfg.grad_width)
        theta, cue = grad.run(cfg.probe_steps, rng)
        h = np.zeros((m, cfg.H))
        A = np.empty((cfg.probe_steps, m))
        cue_prev = 0.0
        for t in range(cfg.probe_steps):
            a, h = batch_step(G, h, cue_prev, cfg.H)
            A[t] = a
            cue_prev = cue[t]
        # per-agent: match quality and predictive info of the action about theta
        d = np.exp(-((A - theta[:, None]) ** 2) / (2.0 * cfg.grad_width**2))
        skill = float(d[500:].mean())
        pis = [predictive_information(A[500:, k], theta[500:]) for k in range(m)]
        brain = float((h ** 2).mean())
        return dict(skill=skill, pred_info=float(np.mean(pis)), brain=brain)


@dataclass
class RunResult:
    history: List[dict]
    final_pop: int
    ended: str
    cfg: EvoConfig


def run(cfg: EvoConfig) -> RunResult:
    pop = Population(cfg)
    probe_rng = np.random.default_rng(cfg.seed + 10_000)
    history = []
    ended = "completed"
    for t in range(1, cfg.ticks + 1):
        pop.step(t)
        if pop.size == 0:
            ended = "extinct"
            history.append(dict(t=t, pop=0))
            break
        if t % cfg.probe_interval == 0:
            pr = pop.probe(probe_rng)
            history.append(dict(t=t, pop=pop.size, mean_gen=float(pop.generation.mean()),
                                mean_Ns=float(pop.Ns.mean()), **pr))
    return RunResult(history=history, final_pop=pop.size, ended=ended, cfg=cfg)
