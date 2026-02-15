"""
Heston model simulation utilities for Monte Carlo / (R)QMC benchmarks.

We implement a robust "full truncation Euler" scheme:
  v_{t+dt} = v_t + kappa*(theta - v^+) dt + xi*sqrt(v^+) dW2
  S_{t+dt} = S_t * exp((r - 0.5*v^+) dt + sqrt(v^+) dW1)

where v^+ = max(v,0), and we clamp v_{t+dt} to be nonnegative.

This is not the most accurate scheme (Andersen QE is better),
but it's stable, simple, and QMC-friendly enough as a first end-to-end case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class HestonParams:
    S0: float = 100.0
    v0: float = 0.04          # initial variance
    r: float = 0.03
    kappa: float = 2.0        # mean reversion speed
    theta: float = 0.04       # long-run variance
    xi: float = 0.5           # vol of vol
    rho: float = -0.7         # corr(dW1, dW2)
    T: float = 1.0
    n_steps: int = 256


def correlate_increments(dW1: np.ndarray, dWperp: np.ndarray, rho: float) -> np.ndarray:
    """
    Build dW2 = rho dW1 + sqrt(1-rho^2) dWperp.
    dW1, dWperp are Brownian increments with the same dt.
    """
    rho = float(rho)
    if rho < -1.0 or rho > 1.0:
        raise ValueError("rho must be in [-1,1]")
    return rho * dW1 + math.sqrt(max(1.0 - rho * rho, 0.0)) * dWperp


def simulate_heston_path_full_trunc(
    p: HestonParams,
    dW1: np.ndarray,
    dWperp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single Heston path using full truncation Euler.

    Inputs:
      dW1:    shape (n_steps,), increments for W1
      dWperp: shape (n_steps,), increments for an independent Brownian motion W⊥

    Returns:
      S_path: shape (n_steps+1,)
      v_path: shape (n_steps+1,)
    """
    n = int(p.n_steps)
    if dW1.shape[0] != n or dWperp.shape[0] != n:
        raise ValueError("dW arrays must have shape (n_steps,)")

    dt = float(p.T) / n

    S = np.empty(n + 1, dtype=float)
    v = np.empty(n + 1, dtype=float)
    S[0] = float(p.S0)
    v[0] = max(float(p.v0), 0.0)

    dW2 = correlate_increments(dW1, dWperp, p.rho)

    for i in range(n):
        v_plus = max(v[i], 0.0)

        # variance update (full truncation Euler)
        v_next = v[i] + p.kappa * (p.theta - v_plus) * dt + p.xi * math.sqrt(v_plus) * float(dW2[i])
        v[i + 1] = max(v_next, 0.0)

        # stock update (log-Euler, exact conditional on variance path)
        drift = (p.r - 0.5 * v_plus) * dt
        diff = math.sqrt(v_plus) * float(dW1[i])
        S[i + 1] = S[i] * math.exp(drift + diff)

    return S, v
