"""
Compute a high-accuracy reference price for a given benchmark JSON "case",
using a strong method at a much larger budget.

Default: RQMC_Sobol(+BB for euler) at N=2^18, R=64.

Output: reference JSON with value, std across replications, and settings.

Usage examples:
  python compute_reference.py --scheme euler --use_bb --N 262144 --R 64 --out ref_heston_asian.json
  python compute_reference.py --scheme qe --N 262144 --R 64 --out ref_heston_asian_qe.json
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
    import numpy as np
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
    import numpy as np
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


def simulate_estimate(N: int, seed: int, scheme: str, use_bb: bool, n_steps: int, params: dict) -> float:
    d = 2 * n_steps
    Z = sobol_normals_base2(N, d, seed=seed)

    # unpack common params
    S0 = params["S0"]; v0 = params["v0"]; r = params["r"]; K = params["K"]
    kappa = params["kappa"]; theta = params["theta"]; xi = params["xi"]; rho = params["rho"]; T = params["T"]
    disc = math.exp(-r * T)

    pay = np.empty(N, dtype=float)
    dt = T / n_steps

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
        return float(np.mean(pay))

    if scheme == "qe":
        p = HestonQEParams(S0=S0, v0=v0, r=r, kappa=kappa, theta=theta, xi=xi, rho=rho, T=T, n_steps=n_steps)
        for i in range(N):
            z_u = Z[i,:n_steps]
            z_ind = Z[i,n_steps:2*n_steps]
            U_var = 0.5 * (1.0 + np.erf(z_u / math.sqrt(2.0)))
            eps = np.finfo(float).eps
            U_var = np.clip(U_var, eps, 1 - eps)
            S_path, _ = simulate_path_qe(p, U_var, z_ind)
            pay[i] = disc * asian_arithmetic_call_payoff(S_path, K)
        return float(np.mean(pay))

    raise ValueError("scheme must be euler or qe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["euler","qe"], default="euler")
    ap.add_argument("--use_bb", action="store_true", help="Only meaningful for scheme=euler")
    ap.add_argument("--n_steps", type=int, default=256)
    ap.add_argument("--N", type=int, default=2**18)
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--seed0", type=int, default=12345)
    ap.add_argument("--out", default="reference_heston_asian.json")

    # Heston + option params
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

    t0 = time.time()
    reps = []
    for rep in range(args.R):
        seed = args.seed0 + 1000 * rep + 17
        reps.append(simulate_estimate(args.N, seed, args.scheme, args.use_bb, args.n_steps, params))
        print(f"[ref] rep {rep+1}/{args.R} done")
    elapsed = time.time() - t0

    mean = float(np.mean(reps))
    std = float(np.std(reps, ddof=1)) if args.R > 1 else 0.0

    out = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "case": "asian_heston",
            "scheme": args.scheme,
            "use_bb": bool(args.use_bb) if args.scheme == "euler" else False,
            "n_steps": args.n_steps,
            "N": args.N,
            "R": args.R,
            "wall_time_sec": float(elapsed),
            "params": params,
        },
        "reference": {"value": mean, "std_across_replications": std, "replications": reps},
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out} | ref={mean:.8f} std={std:.8f}")


if __name__ == "__main__":
    main()
