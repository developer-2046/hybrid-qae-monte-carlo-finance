"""
Heston model simulation (risk-neutral) using Full Truncation Euler for variance,
and log-Euler for the asset.

This is a stable baseline scheme, but for publication-grade discretization you
should prefer Andersen's Quadratic-Exponential (QE) scheme (see heston_qe.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class HestonParams:
    S0: float = 100.0
    v0: float = 0.04
    r: float = 0.03
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7
    T: float = 1.0
    n_steps: int = 256


def correlate_increments(dW1: np.ndarray, dWperp: np.ndarray, rho: float) -> np.ndarray:
    """dW2 = rho dW1 + sqrt(1-rho^2) dWperp."""
    rho = float(rho)
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be in [-1,1]")
    return rho * dW1 + math.sqrt(max(1.0 - rho * rho, 0.0)) * dWperp


def simulate_path_full_trunc_euler(
    p: HestonParams,
    dW1: np.ndarray,
    dWperp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single Heston path.

    Inputs:
      dW1:    (n_steps,) Brownian increments for asset driver W1
      dWperp: (n_steps,) independent Brownian increments for W_perp

    Returns:
      S_path: (n_steps+1,)
      v_path: (n_steps+1,)
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

        # Full truncation Euler for v
        v_next = v[i] + p.kappa * (p.theta - v_plus) * dt + p.xi * math.sqrt(v_plus) * float(dW2[i])
        v[i + 1] = max(v_next, 0.0)

        # Log-Euler for S conditional on v_plus
        drift = (p.r - 0.5 * v_plus) * dt
        diff = math.sqrt(v_plus) * float(dW1[i])
        S[i + 1] = S[i] * math.exp(drift + diff)

    return S, v
