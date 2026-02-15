"""
End-to-end benchmark for Heston Asian (arithmetic) option pricing.

Supports two discretizations:
  - euler: Full truncation Euler (variance) + log-Euler (asset)  [BB supported]
  - qe:    Andersen QE for variance + trapezoid I, analytic corr term [BB NOT used]

Outputs a JSON compatible with plot_error_cost_v2.py.
"""

import math
import time
import json
import argparse
from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.stats import norm
from scipy.stats import qmc

from models.heston_euler import HestonParams as HestonEulerParams, simulate_path_full_trunc_euler
from models.heston_qe import HestonParams as HestonQEParams, simulate_path_qe


def asian_arithmetic_call_payoff(S_path: np.ndarray, K: float) -> float:
    A = float(np.mean(S_path[1:]))
    return max(A - K, 0.0)


# ----------------------------
# Brownian bridge (for Euler mode only)
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
        raise ValueError("n must be power of 2 for Sobol random_base2().")
    eng = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = eng.random_base2(m=m)
    eps = np.finfo(float).eps
    u = np.clip(u, eps, 1 - eps)
    return norm.ppf(u)


@dataclass
class CaseConfig:
    name: str = "asian_heston"
    K: float = 100.0
    # Heston params
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


def simulate_once(cfg: CaseConfig, Z: np.ndarray, scheme: str, use_bb: bool) -> float:
    n = cfg.n_steps
    disc = math.exp(-cfg.r * cfg.T)

    if scheme == "euler":
        # Z: (2n,) normals -> dW1 and dWperp
        dt = cfg.T / n
        z1 = Z[:n]
        zperp = Z[n:2*n]

        if use_bb:
            W1 = brownian_bridge_W(cfg.T, n, z1)
            Wp = brownian_bridge_W(cfg.T, n, zperp)
            dW1 = increments_from_W(W1)
            dWp = increments_from_W(Wp)
        else:
            dW1 = math.sqrt(dt) * z1
            dWp = math.sqrt(dt) * zperp

        p = HestonEulerParams(
            S0=cfg.S0, v0=cfg.v0, r=cfg.r,
            kappa=cfg.kappa, theta=cfg.theta, xi=cfg.xi, rho=cfg.rho,
            T=cfg.T, n_steps=cfg.n_steps
        )
        S_path, _ = simulate_path_full_trunc_euler(p, dW1, dWp)
        return disc * asian_arithmetic_call_payoff(S_path, cfg.K)

    if scheme == "qe":
        # Z: (2n,) normals -> U for variance + Z for independent price shock
        # Convert first block to uniforms using Phi
        z_u = Z[:n]
        z_ind = Z[n:2*n]
        U_var = norm.cdf(z_u)
        # Guard against 0/1 exactly
        eps = np.finfo(float).eps
        U_var = np.clip(U_var, eps, 1 - eps)

        p = HestonQEParams(
            S0=cfg.S0, v0=cfg.v0, r=cfg.r,
            kappa=cfg.kappa, theta=cfg.theta, xi=cfg.xi, rho=cfg.rho,
            T=cfg.T, n_steps=cfg.n_steps
        )
        S_path, _ = simulate_path_qe(p, U_var, z_ind)
        return disc * asian_arithmetic_call_payoff(S_path, cfg.K)

    raise ValueError("scheme must be one of: euler, qe")


def simulate_case_once(cfg: CaseConfig, Z: np.ndarray, scheme: str, use_bb: bool, antithetic: bool) -> float:
    if not antithetic:
        return simulate_once(cfg, Z, scheme, use_bb)
    return 0.5 * (simulate_once(cfg, Z, scheme, use_bb) + simulate_once(cfg, -Z, scheme, use_bb))


def run_method(cfg: CaseConfig, method: str, N: int, R: int, scheme: str, use_bb: bool, antithetic: bool) -> Dict:
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
            payoffs[i] = simulate_case_once(cfg, Z[i], scheme, use_bb, antithetic)
        rep_est[rep] = float(np.mean(payoffs))

    elapsed = time.time() - t0

    tag = f"{method}"
    if antithetic:
        tag += "+AV"
    if scheme == "euler" and use_bb:
        tag += "+BB"
    tag += f"+{scheme}"

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
            "scheme": scheme,
            "n_steps": cfg.n_steps,
            "use_bb": bool(use_bb) if scheme == "euler" else False,
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["euler", "qe"], default="euler")
    ap.add_argument("--n_steps", type=int, default=256)
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--N_max_pow", type=int, default=14, help="max power for N=2^k, starting at k=8")
    ap.add_argument("--out", default="bench_results_heston_asian.json")
    args = ap.parse_args()

    Ns = [2**k for k in range(8, args.N_max_pow + 1)]
    cfg = CaseConfig(n_steps=args.n_steps)

    results = []
    for N in Ns:
        # MC
        results.append(run_method(cfg, "MC", N, args.R, args.scheme, use_bb=False, antithetic=False))
        results.append(run_method(cfg, "MC", N, args.R, args.scheme, use_bb=False, antithetic=True))

        # RQMC
        results.append(run_method(cfg, "RQMC_Sobol", N, args.R, args.scheme, use_bb=False, antithetic=False))
        if args.scheme == "euler":
            results.append(run_method(cfg, "RQMC_Sobol", N, args.R, args.scheme, use_bb=True, antithetic=False))

        print(f"[done] {cfg.name} N={N} scheme={args.scheme}")

    out = {
        "meta": {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "Ns": Ns, "R": args.R, "scheme": args.scheme, "n_steps": args.n_steps},
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
