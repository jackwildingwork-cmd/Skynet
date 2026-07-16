"""Fixed controllers for the milestone: a reactive one and a predictive one.

Both must choose the coupling action a(t) for the upcoming gradient type theta(t)
using only cues from the past. The difference is whether they carry an internal
MODEL of the environment's dynamics.

  Reactive  : a(t) = c(t-1). Uses the last cue directly; no memory, no model of
              rho, no filtering of cue noise.
  Predictive: maintains a Kalman estimate of theta under the known AR(1)+noise
              model and predicts a(t) = E[theta(t) | cues up to t-1] = rho * xhat.
              This is the optimal predictor; its internal state xhat is the
              "model". It costs extra to maintain (charged in the agent layer).

The milestone question: does the predictive controller capture more free energy,
and does that advantage vanish when the environment is unpredictable (rho small
or cue noise large)? If not, "intelligence evolves" has no thermodynamic basis.
"""

from __future__ import annotations

import numpy as np

from .environment import Gradient


def reactive_capture(grad: Gradient, theta: np.ndarray, cue: np.ndarray) -> np.ndarray:
    """a(t) = c(t-1); a(0)=0."""
    a = np.empty_like(theta)
    a[0] = 0.0
    a[1:] = cue[:-1]
    return grad.capture(a, theta)


def predictive_capture(grad: Gradient, theta: np.ndarray, cue: np.ndarray):
    """Kalman filter for AR(1) theta with noisy observations; predict one step.

    Returns (capture, xhat_pred) where xhat_pred[t] = E[theta(t)|cues<t], the
    action taken, and also the estimate used (for measuring predictive info)."""
    rho = grad.rho
    R = grad.cue_noise**2          # observation noise variance
    steps = len(theta)
    xhat = 0.0                     # posterior mean of theta(t-1)
    P = 1.0                        # posterior variance
    a = np.empty(steps)
    xpred = np.empty(steps)
    for t in range(steps):
        # predict theta(t) from posterior at t-1
        x_pred = rho * xhat
        P_pred = rho**2 * P + (1.0 - rho**2)
        a[t] = x_pred
        xpred[t] = x_pred
        # observe cue(t) AFTER acting, update posterior for next step
        if R > 0:
            K = P_pred / (P_pred + R)
        else:
            K = 1.0
        xhat = x_pred + K * (cue[t] - x_pred)
        P = (1.0 - K) * P_pred
    return grad.capture(a, theta), xpred


def predictive_information(xpred: np.ndarray, theta: np.ndarray) -> float:
    """I(prediction; theta) for jointly-Gaussian variables:
    I = -0.5 log(1 - corr^2), in nats. The predictive information the agent's
    internal state holds about the gradient it is about to face."""
    if len(theta) < 3 or np.std(xpred) < 1e-12 or np.std(theta) < 1e-12:
        return 0.0
    c = np.corrcoef(xpred, theta)[0, 1]
    c = max(-0.999999, min(0.999999, 0.0 if np.isnan(c) else c))
    return -0.5 * float(np.log(1.0 - c**2))
