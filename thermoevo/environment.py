"""A structured, fluctuating gradient — the environment that can reward prediction.

The gradient has a hidden "type" theta(t) the agent must match its coupling to in
order to capture free energy. theta is an AR(1) process with correlation rho
(the environment's PREDICTABILITY knob): rho -> 1 is highly predictable, rho -> 0
is white noise. The agent must commit its coupling action a(t) using only
information available BEFORE theta(t) is revealed (a noisy cue of the past), so an
agent that models theta can pre-position and capture more.

Captured power:  P_in(t) = G0 * exp(-(a(t) - theta(t))^2 / (2 w^2)).

This is the concrete realisation of the thermodynamics-of-prediction result: the
extra free energy a predictive agent captures over a reactive one is bounded by
the predictive information its internal state holds about the future gradient, and
that information is available only to the extent the environment is predictable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Gradient:
    rho: float           # AR(1) correlation = predictability (0=white, ->1=persistent)
    cue_noise: float     # sigma_c: observation noise on the cue (0=perfect cue)
    G0: float = 1.0      # peak capturable power
    width: float = 1.0   # w: coupling-match tolerance

    def run(self, steps: int, rng: np.random.Generator):
        """Generate theta(t) and noisy cues c(t)=theta(t)+noise."""
        theta = np.empty(steps)
        theta[0] = rng.standard_normal()
        s = math.sqrt(max(0.0, 1.0 - self.rho**2))
        for t in range(1, steps):
            theta[t] = self.rho * theta[t - 1] + s * rng.standard_normal()
        cue = theta + self.cue_noise * rng.standard_normal(steps)
        return theta, cue

    def capture(self, a: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self.G0 * np.exp(-((a - theta) ** 2) / (2.0 * self.width**2))
