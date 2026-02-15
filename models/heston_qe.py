"""
Heston model simulation using Andersen's Quadratic-Exponential (QE) scheme
for the variance process.

References:
  - L. Andersen (2008), "Simple and Efficient Simulation of the Heston Stochastic Volatility Model"

Notes:
- This implementation uses QE for v_{t+dt} and a trapezoidal approximation for
  integrated variance I = \int_t^{t+dt} v_s ds as I ≈ 0.5*(v_t + v_{t+dt})*dt.
  This is a common, robust "QE-M" style variant.
- The log-price update uses the standard decomposition:
    log S_{t+dt} = log S_t + r dt - 0.5 I
                  + (rho/xi) * (v_{t+dt} - v_t - kappa*theta*dt + kappa*I)
                  + sqrt((1-rho^2)*I) * Z
  where Z ~ N(0,1) independent of the variance draw.
- Randomness per step: one U(0,1) for the QE variance draw, and one N(0,1)
  for the independent price shock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.stats import norm


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


def _qe_variance_step(v: float, p: HestonParams, dt: float, u: float, psi_c: float = 1.5) -> float:
    """
    One QE update for v_{t+dt} given v_t=v and uniform u in (0,1).
    """
    kappa, theta, xi = float(p.kappa), float(p.theta), float(p.xi)
    v = max(float(v), 0.0)

    exp_kdt = math.exp(-kappa * dt)
    m = theta + (v - theta) * exp_kdt

    s2 = (v * xi * xi * exp_kdt * (1 - exp_kdt) / kappa) + (theta * xi * xi * (1 - exp_kdt) ** 2 / (2 * kappa))
    if s2 < 0.0:
        s2 = 0.0

    # Avoid division issues when m is ~0
    if m <= 1e-16:
        return 0.0

    psi = s2 / (m * m) if m > 0 else float("inf")

    if psi <= psi_c:
        # Quadratic-Gaussian regime
        b2 = 2.0 / psi - 1.0 + math.sqrt(max(2.0 / psi, 0.0)) * math.sqrt(max(2.0 / psi - 1.0, 0.0))
        b = math.sqrt(max(b2, 0.0))
        a = m / (1.0 + b2)

        # Convert u to standard normal
        z = norm.ppf(min(max(u, 1e-15), 1 - 1e-15))
        v_next = a * (b + z) ** 2
        return max(v_next, 0.0)

    # Exponential regime
    p0 = (psi - 1.0) / (psi + 1.0)  # mass at zero
    beta = (1.0 - p0) / m
    if u <= p0:
        return 0.0
    # Inverse CDF of shifted exponential
    v_next = -math.log((1.0 - u) / (1.0 - p0)) / beta
    return max(v_next, 0.0)


def simulate_path_qe(
    p: HestonParams,
    U_var: np.ndarray,
    Z_indep: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single Heston path with QE variance scheme.

    Inputs:
      U_var:   (n_steps,) uniforms in (0,1) for variance draw
      Z_indep: (n_steps,) standard normals for independent log-price shock

    Returns:
      S_path: (n_steps+1,)
      v_path: (n_steps+1,)
    """
    n = int(p.n_steps)
    if U_var.shape[0] != n or Z_indep.shape[0] != n:
        raise ValueError("U_var and Z_indep must have shape (n_steps,)")

    dt = float(p.T) / n

    S = np.empty(n + 1, dtype=float)
    v = np.empty(n + 1, dtype=float)

    S[0] = float(p.S0)
    v[0] = max(float(p.v0), 0.0)
    logS = math.log(S[0])

    rho = float(p.rho)
    xi = float(p.xi)
    kappa = float(p.kappa)
    theta = float(p.theta)

    for i in range(n):
        v_t = v[i]
        v_next = _qe_variance_step(v_t, p, dt, float(U_var[i]))
        v[i + 1] = v_next

        # Integrated variance (trapezoid)
        I = 0.5 * (v_t + v_next) * dt
        I = max(I, 0.0)

        # Drift-correction term for correlation (Andersen)
        corr_term = 0.0
        if xi > 0.0:
            corr_term = (rho / xi) * (v_next - v_t - kappa * theta * dt + kappa * I)

        # Independent diffusion term
        diff = math.sqrt(max((1.0 - rho * rho) * I, 0.0)) * float(Z_indep[i])

        logS = logS + p.r * dt - 0.5 * I + corr_term + diff
        S[i + 1] = math.exp(logS)

    return S, v
