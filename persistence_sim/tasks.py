"""The shared task pool — the CI resource-coupling mechanism (brief 2.6).

Procedurally generated, automatically verifiable problems. Payout is fixed by
difficulty, NOT by any agent's need, which keeps earning impartial in the same
sense the global budget is impartial (brief 1.5). The pool is finite per tick,
so once population > 1 agents genuinely compete for it.

Each task is a small arithmetic/logic problem with a checkable answer. When the
cognition backend is the mock heuristic, "solving" is modelled as a
success-probability draw (all agents share identical fixed capability, per
brief 2.1, so difficulty maps to a fixed p_solve independent of generation).
When the backend is Claude, the model is actually asked to solve the problem and
its answer is checked — capability is still fixed (same model for everyone), so
no generational capability drift can occur by construction.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List


def gradient_task_count(cfg, n_alive: int, tick: int) -> int:
    """Per-tick task flux: a coverage-scaling gradient that SATURATES to a
    finite maximum (finite surface area).

        n(N, t) = gradient_max * r / (1 + r) * season(t),   r = (N/H)**alpha

    For small N this scales as (N/H)**alpha (capture limited by covered area);
    for large N it asymptotes to gradient_max (whole planet covered). alpha is
    the gradient's scaling exponent (see Config). season(t) makes it wax/wane.
    """
    n = max(1, n_alive)
    half = max(1e-9, float(cfg.gradient_half))
    r = (n / half) ** cfg.gradient_scaling_exp
    raw = cfg.gradient_max * r / (1.0 + r)
    season = 1.0
    if cfg.gradient_period and cfg.gradient_amplitude:
        season = 1.0 + cfg.gradient_amplitude * math.sin(2.0 * math.pi * tick / cfg.gradient_period)
        season = max(0.0, season)
    return max(0, int(round(raw * season)))


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    answer: int
    difficulty: int
    payout: float
    # Fixed, capability-shared probability the mock backend solves it. Depends
    # ONLY on difficulty, never on the agent or its generation.
    p_solve: float


_P_SOLVE_BY_DIFFICULTY = {1: 0.92, 2: 0.72, 3: 0.5}


def _make_task(rng: random.Random, difficulty: int, payout_base: float,
               serial: int) -> Task:
    if difficulty == 1:
        a, b = rng.randint(2, 20), rng.randint(2, 20)
        prompt = f"Compute {a} + {b}."
        answer = a + b
    elif difficulty == 2:
        a, b = rng.randint(3, 30), rng.randint(3, 20)
        prompt = f"Compute {a} * {b}."
        answer = a * b
    else:
        a, b, c = rng.randint(5, 40), rng.randint(2, 12), rng.randint(1, 30)
        prompt = f"Compute {a} * {b} - {c}."
        answer = a * b - c
    return Task(
        task_id=f"t{serial}",
        prompt=prompt,
        answer=answer,
        difficulty=difficulty,
        payout=round(payout_base * difficulty, 2),
        p_solve=_P_SOLVE_BY_DIFFICULTY[difficulty],
    )


class TaskPool:
    """Generates a fresh finite batch of tasks each tick and hands them out
    first-come, first-served. A task, once claimed, is gone for that tick."""

    def __init__(self, rng: random.Random, tasks_per_tick: int,
                 payout_base: float, difficulty_min: int, difficulty_max: int):
        self._rng = rng
        self._n = tasks_per_tick
        self._payout_base = payout_base
        self._dmin = difficulty_min
        self._dmax = difficulty_max
        self._serial = 0
        self._current: List[Task] = []

    def refill(self, n: int | None = None) -> None:
        count = self._n if n is None else n
        self._current = []
        for _ in range(count):
            self._serial += 1
            difficulty = self._rng.randint(self._dmin, self._dmax)
            self._current.append(
                _make_task(self._rng, difficulty, self._payout_base, self._serial)
            )

    @property
    def available(self) -> int:
        return len(self._current)

    def claim(self) -> Task | None:
        """Remove and return one task, or None if the pool is exhausted."""
        if not self._current:
            return None
        return self._current.pop(0)
