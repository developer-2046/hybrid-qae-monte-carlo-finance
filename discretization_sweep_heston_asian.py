"""
Discretization sweep for Heston Asian call.

Goal: separate time-discretization error from sampling error by comparing prices
across n_steps values under a fixed strong estimator (default RQMC_Sobol+BB+euler).

It writes a JSON file with a list of (n_steps, estimate_mean, estimate_std, cost, wall_time).

Example:
  python discretization_sweep_heston_asian.py --scheme euler --use_bb --N 16384 --R 32 --out disc_sweep.json
  python discretization_sweep_heston_asian.py --scheme qe --N 16384 --R 32 --out disc_sweep_qe.json
"""

import math
import time
import json
import argparse
import numpy as np
from scipy.stats import norm
from scipy.stats import qmc

from models.heston_euler import HestonParams as HestonEulerParams, simulate_path_full_trunc_euler
from models.heston_qe import HestonParams as HestonQEParams, simulate_path_qe


def asian_arithmetic_call_payoff(S_path: np.ndarray, K: float) -> float:
    return max(float(np.mean(S_path[1:])) - K, 0.0)


def brownian_bridge_W(T: float, n_steps: int, z: np.ndarray) -> np.ndarray:
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


def sobol_normals_base2(n: int, d: int, seed: int) -> np.ndarray:
    m = int(round(math.log2(n)))
    if 2**m != n:
        raise ValueError("n must be power of 2 for Sobol random_base2().")
    eng = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = eng.random_base2(m=m)
    eps = np.finfo(float).eps
    u = np.clip(u, eps, 1 - eps)
    return norm.ppf(u)


def estimate_price(n_steps: int, scheme: str, use_bb: bool, N: int, R: int, seed0: int, params: dict):
    d = 2 * n_steps
    reps = []
    start = time.time()

    S0 = params["S0"]; v0 = params["v0"]; r = params["r"]; K = params["K"]
    kappa = params["kappa"]; theta = params["theta"]; xi = params["xi"]; rho = params["rho"]; T = params["T"]
    disc = math.exp(-r * T)
    dt = T / n_steps

    for rep in range(R):
        seed = seed0 + 1000 * rep + 17
        Z = sobol_normals_base2(N, d, seed=seed)

        pay = np.empty(N, dtype=float)

        if scheme == "euler":
            p = HestonEulerParams(S0=S0, v0=v0, r=r, kappa=kappa, theta=theta, xi=xi, rho=rho, T=T, n_steps=n_steps)
            for i in range(N):
                z1 = Z[i,:n_steps]
                zp = Z[i,n_steps:2*n_steps]
                if use_bb:
                    W1 = brownian_bridge_W(T, n_steps, z1)
                    Wp = brownian_bridge_W(T, n_steps, zp)
                    dW1 = increments_from_W(W1)
                    dWp = increments_from_W(Wp)
                else:
                    dW1 = math.sqrt(dt) * z1
                    dWp = math.sqrt(dt) * zp
                S_path, _ = simulate_path_full_trunc_euler(p, dW1, dWp)
                pay[i] = disc * asian_arithmetic_call_payoff(S_path, K)
        else:
            p = HestonQEParams(S0=S0, v0=v0, r=r, kappa=kappa, theta=theta, xi=xi, rho=rho, T=T, n_steps=n_steps)
            for i in range(N):
                z_u = Z[i,:n_steps]
                z_ind = Z[i,n_steps:2*n_steps]
                U_var = 0.5 * (1.0 + np.erf(z_u / math.sqrt(2.0)))
                eps = np.finfo(float).eps
                U_var = np.clip(U_var, eps, 1 - eps)
                S_path, _ = simulate_path_qe(p, U_var, z_ind)
                pay[i] = disc * asian_arithmetic_call_payoff(S_path, K)

        reps.append(float(np.mean(pay)))
        print(f"[sweep] n_steps={n_steps} rep {rep+1}/{R} done")

    elapsed = time.time() - start
    mean = float(np.mean(reps))
    std = float(np.std(reps, ddof=1)) if R > 1 else 0.0
    return mean, std, reps, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["euler","qe"], default="euler")
    ap.add_argument("--use_bb", action="store_true")
    ap.add_argument("--N", type=int, default=16384)
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=12345)
    ap.add_argument("--out", default="disc_sweep_heston_asian.json")
    ap.add_argument("--steps", type=str, default="64,128,256,512")

    # params
    ap.add_argument("--S0", type=float, default=100.0)
    ap.add_argument("--v0", type=float, default=0.04)
    ap.add_argument("--r", type=float, default=0.03)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--theta", type=float, default=0.04)
    ap.add_argument("--xi", type=float, default=0.5)
    ap.add_argument("--rho", type=float, default=-0.7)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--K", type=float, default=100.0)

    args = ap.parse_args()

    params = dict(S0=args.S0, v0=args.v0, r=args.r, kappa=args.kappa, theta=args.theta, xi=args.xi,
                  rho=args.rho, T=args.T, K=args.K)

    sweep_steps = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    rows = []
    for n_steps in sweep_steps:
        mean, std, reps, elapsed = estimate_price(n_steps, args.scheme, args.use_bb, args.N, args.R, args.seed0, params)
        rows.append({
            "n_steps": n_steps,
            "estimate_mean": mean,
            "estimate_std": std,
            "replications": reps,
            "cost_payoff_evals": int(args.N * args.R),
            "wall_time_sec": float(elapsed),
        })

    out = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "case": "asian_heston",
            "scheme": args.scheme,
            "use_bb": bool(args.use_bb) if args.scheme == "euler" else False,
            "N": args.N,
            "R": args.R,
            "steps": sweep_steps,
            "params": params
        },
        "rows": rows
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
