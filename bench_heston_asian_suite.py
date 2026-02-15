import math
import time
import json
from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.stats import norm
from scipy.stats import qmc

from models.heston import HestonParams, simulate_heston_path_full_trunc


# ----------------------------
# Payoff: Asian arithmetic-average call
# ----------------------------
def asian_arithmetic_call_payoff(S_path: np.ndarray, K: float) -> float:
    A = float(np.mean(S_path[1:]))  # average over monitoring times (exclude t=0)
    return max(A - K, 0.0)


# ----------------------------
# Brownian bridge construction (dimension reduction)
# ----------------------------
def brownian_bridge_W(T: float, n_steps: int, z: np.ndarray) -> np.ndarray:
    if z.shape[0] != n_steps:
        raise ValueError("z must have length n_steps")

    t = np.linspace(0.0, T, n_steps + 1)
    W = np.full(n_steps + 1, np.nan, dtype=float)
    W[0] = 0.0
    W[n_steps] = math.sqrt(T) * float(z[0])

    k = 1
    queue = [(0, n_steps)]
    while queue:
        left, right = queue.pop(0)
        if right - left <= 1:
            continue
        mid = (left + right) // 2
        if not np.isnan(W[mid]):
            continue

        tl, tm, tr = t[left], t[mid], t[right]
        mean = ((tr - tm) / (tr - tl)) * W[left] + ((tm - tl) / (tr - tl)) * W[right]
        var = (tm - tl) * (tr - tm) / (tr - tl)
        W[mid] = mean + math.sqrt(max(var, 0.0)) * float(z[k])
        k += 1

        queue.append((left, mid))
        queue.append((mid, right))

    nan_idx = np.where(np.isnan(W))[0]
    for idx in nan_idx:
        W[idx] = math.sqrt(t[idx]) * float(z[min(k, n_steps - 1)])
        k += 1

    return W


def increments_from_W(W: np.ndarray) -> np.ndarray:
    return np.diff(W)


# ----------------------------
# RQMC generator
# ----------------------------
def sobol_normals_base2(n: int, d: int, seed: int) -> np.ndarray:
    m = int(round(math.log2(n)))
    if 2**m != n:
        raise ValueError("n must be a power of 2 for random_base2().")

    eng = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = eng.random_base2(m=m)

    eps = np.finfo(float).eps
    u = np.clip(u, eps, 1 - eps)
    return norm.ppf(u)


# ----------------------------
# Benchmark harness (Heston Asian)
# ----------------------------
@dataclass
class CaseConfig:
    name: str = "asian_heston"
    # option params
    K: float = 100.0
    # heston params
    S0: float = 100.0
    v0: float = 0.04
    r: float = 0.03
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7
    T: float = 1.0
    n_steps: int = 256
    seed0: int = 12345


def simulate_once(cfg: CaseConfig, Z: np.ndarray, use_brownian_bridge: bool) -> float:
    """
    Z shape: (2*n_steps,) standard normals:
      first n_steps -> W1 driver
      next  n_steps -> independent driver W⊥
    """
    n = cfg.n_steps
    dt = cfg.T / n

    z1 = Z[:n]
    zperp = Z[n:2*n]

    if use_brownian_bridge:
        W1 = brownian_bridge_W(cfg.T, n, z1)
        Wp = brownian_bridge_W(cfg.T, n, zperp)
        dW1 = increments_from_W(W1)
        dWp = increments_from_W(Wp)
    else:
        dW1 = math.sqrt(dt) * z1
        dWp = math.sqrt(dt) * zperp

    p = HestonParams(
        S0=cfg.S0,
        v0=cfg.v0,
        r=cfg.r,
        kappa=cfg.kappa,
        theta=cfg.theta,
        xi=cfg.xi,
        rho=cfg.rho,
        T=cfg.T,
        n_steps=cfg.n_steps,
    )

    S_path, _v_path = simulate_heston_path_full_trunc(p, dW1, dWp)
    disc = math.exp(-cfg.r * cfg.T)
    return disc * asian_arithmetic_call_payoff(S_path, cfg.K)


def simulate_case_once(cfg: CaseConfig, Z: np.ndarray, use_brownian_bridge: bool, antithetic: bool) -> float:
    if not antithetic:
        return simulate_once(cfg, Z, use_brownian_bridge)
    return 0.5 * (simulate_once(cfg, Z, use_brownian_bridge) + simulate_once(cfg, -Z, use_brownian_bridge))


def run_method(cfg: CaseConfig, method: str, N: int, R: int, use_brownian_bridge: bool, antithetic: bool) -> Dict:
    """
    Runs R replications; each replication produces an estimator using N samples.
    MC: fresh RNG seed per replication.
    RQMC_Sobol: fresh scramble seed per replication.
    """
    t0 = time.time()
    d = 2 * cfg.n_steps
    rep_est = np.empty(R, dtype=float)

    for rep in range(R):
        seed = cfg.seed0 + 1000 * rep + 17

        if method == "MC":
            rng = np.random.default_rng(seed)
            Z = rng.standard_normal(size=(N, d))
        elif method == "RQMC_Sobol":
            Z = sobol_normals_base2(N, d, seed=seed)
        else:
            raise ValueError(f"Unknown method: {method}")

        payoffs = np.empty(N, dtype=float)
        for i in range(N):
            payoffs[i] = simulate_case_once(cfg, Z[i], use_brownian_bridge, antithetic)
        rep_est[rep] = float(np.mean(payoffs))

    elapsed = time.time() - t0

    tag = method
    if use_brownian_bridge:
        tag += "+BB"
    if antithetic:
        tag += "+AV"

    return {
        "case": cfg.name,
        "method": tag,
        "N": N,
        "R": R,
        "estimate_mean": float(np.mean(rep_est)),
        "estimate_std": float(np.std(rep_est, ddof=1)) if R > 1 else 0.0,
        "replications": rep_est.tolist(),
        "cost_payoff_evals": int(N * R),
        "wall_time_sec": float(elapsed),
        "notes": {
            "model": "Heston",
            "scheme": "full_truncation_euler",
            "n_steps": cfg.n_steps,
            "params": {
                "S0": cfg.S0, "v0": cfg.v0, "r": cfg.r,
                "kappa": cfg.kappa, "theta": cfg.theta, "xi": cfg.xi, "rho": cfg.rho,
                "T": cfg.T, "K": cfg.K
            }
        }
    }


def main():
    Ns = [2**k for k in range(8, 15)]  # 256 .. 16384
    R = 32

    cfg = CaseConfig()

    results = []
    for N in Ns:
        # MC baselines
        results.append(run_method(cfg, "MC", N, R, use_brownian_bridge=False, antithetic=False))
        results.append(run_method(cfg, "MC", N, R, use_brownian_bridge=False, antithetic=True))

        # RQMC baselines
        results.append(run_method(cfg, "RQMC_Sobol", N, R, use_brownian_bridge=False, antithetic=False))
        results.append(run_method(cfg, "RQMC_Sobol", N, R, use_brownian_bridge=True, antithetic=False))

        print(f"[done] {cfg.name} N={N}")

    out = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Ns": Ns,
            "R": R,
            "case": cfg.name,
            "notes": "Heston Asian (arithmetic) via full truncation Euler. RQMC uses scrambled Sobol, optionally with BB on both W1 and W_perp."
        },
        "results": results
    }
    with open("bench_results_heston_asian.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote bench_results_heston_asian.json")


if __name__ == "__main__":
    main()
