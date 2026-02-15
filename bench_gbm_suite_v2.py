import math
import time
import json
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.stats import norm
from scipy.stats import qmc

# ----------------------------
# Payoffs
# ----------------------------
def asian_arithmetic_call_payoff(S_path: np.ndarray, K: float) -> float:
    "Arithmetic-average Asian call. S_path shape: (n_steps+1,)."
    A = float(np.mean(S_path[1:]))  # average over monitoring times (exclude t=0)
    return max(A - K, 0.0)


def basket_call_payoff(S_T: np.ndarray, K: float, weights: Optional[np.ndarray] = None) -> float:
    "Basket call on terminal prices S_T shape: (d_assets,)."
    if weights is None:
        weights = np.ones_like(S_T, dtype=float) / len(S_T)
    basket = float(np.dot(weights, S_T))
    return max(basket - K, 0.0)


# ----------------------------
# GBM simulation utilities
# ----------------------------
def gbm_path_from_increments(S0: float, r: float, sigma: float, T: float, dW: np.ndarray) -> np.ndarray:
    # Exact GBM update at monitoring times using log-increments.
    # dW: shape (n_steps,), Brownian increments ~ N(0, dt)
    n_steps = int(dW.shape[0])
    dt = T / n_steps
    S = np.empty(n_steps + 1, dtype=float)
    S[0] = S0

    drift = (r - 0.5 * sigma * sigma) * dt
    logS = math.log(S0)
    for i in range(n_steps):
        logS = logS + drift + sigma * dW[i]
        S[i + 1] = math.exp(logS)
    return S


# ----------------------------
# Brownian bridge construction (dimension reduction)
# ----------------------------
def brownian_bridge_W(T: float, n_steps: int, z: np.ndarray) -> np.ndarray:
    # Construct Brownian motion values W(t_i) at i=0..n_steps using midpoint-recursion Brownian bridge.
    # Requires z of length n_steps (standard normals).
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

    # Safety fill if any NaNs remain
    nan_idx = np.where(np.isnan(W))[0]
    for idx in nan_idx:
        W[idx] = math.sqrt(t[idx]) * float(z[min(k, n_steps - 1)])
        k += 1

    return W


def increments_from_W(W: np.ndarray) -> np.ndarray:
    return np.diff(W)


# ----------------------------
# RQMC generators
# ----------------------------
def sobol_normals_base2(n: int, d: int, seed: int) -> np.ndarray:
    # Generate n=2^m scrambled Sobol points in [0,1)^d (SciPy: LMS+shift scrambling),
    # map to N(0,1) via inverse CDF.
    m = int(round(math.log2(n)))
    if 2**m != n:
        raise ValueError("n must be a power of 2 for random_base2().")

    eng = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = eng.random_base2(m=m)

    eps = np.finfo(float).eps
    u = np.clip(u, eps, 1 - eps)
    return norm.ppf(u)


# ----------------------------
# Barrier: continuous monitoring correction via Brownian bridge in log-space
# (Variance-reduced / QMC-friendly conditional weighting)
# ----------------------------
def up_and_out_survival_prob_log_bridge(x0: float, x1: float, logB: float, sigma: float, dt: float) -> float:
    # If endpoints are below logB, survival prob = 1 - exp(-2 (logB-x0)(logB-x1) / (sigma^2 dt)).
    if x0 >= logB or x1 >= logB:
        return 0.0
    denom = (sigma * sigma) * dt
    if denom <= 0.0:
        return 1.0
    p_cross = math.exp(-2.0 * (logB - x0) * (logB - x1) / denom)
    # numeric guards
    p_cross = min(max(p_cross, 0.0), 1.0)
    return 1.0 - p_cross


def barrier_uo_call_continuous_weighted(S_path: np.ndarray, K: float, B: float, sigma: float, dt: float) -> float:
    # Up-and-out barrier call with continuous monitoring correction (per-step survival weights).
    if np.any(S_path >= B):
        return 0.0

    logB = math.log(B)
    logS = np.log(S_path)

    survival_weight = 1.0
    n_steps = len(S_path) - 1
    for i in range(n_steps):
        survival_weight *= up_and_out_survival_prob_log_bridge(logS[i], logS[i + 1], logB, sigma, dt)
        if survival_weight == 0.0:
            return 0.0

    return max(S_path[-1] - K, 0.0) * survival_weight


# ----------------------------
# Experiment harness
# ----------------------------
@dataclass
class CaseConfig:
    name: str
    S0: float = 100.0
    K: float = 100.0
    r: float = 0.03
    sigma: float = 0.2
    T: float = 1.0
    n_steps: int = 256      # power of 2 helps QMC + Brownian bridge
    barrier_B: float = 130.0
    d_assets: int = 16
    rho: float = 0.3        # equicorrelation target for basket
    seed0: int = 12345


def basket_cholesky(d: int, rho: float) -> np.ndarray:
    # Cholesky factor for equicorrelation matrix (1 on diag, rho off diag).
    if d == 1:
        return np.ones((1, 1), dtype=float)
    if rho <= -1.0 / (d - 1):
        raise ValueError(f"rho={rho} not PSD for d={d}. Need rho > -1/(d-1).")
    Sigma = np.full((d, d), rho, dtype=float)
    np.fill_diagonal(Sigma, 1.0)
    return np.linalg.cholesky(Sigma)


def simulate_case_once(
    cfg: CaseConfig,
    Z: np.ndarray,
    use_brownian_bridge: bool,
    antithetic: bool,
    barrier_continuous_weighting: bool,
) -> float:
    # Returns discounted payoff for one sample (or averaged antithetic pair).
    disc = math.exp(-cfg.r * cfg.T)

    def one_payoff(z: np.ndarray) -> float:
        if cfg.name in ("asian_gbm", "barrier_uo_gbm"):
            if use_brownian_bridge:
                W = brownian_bridge_W(cfg.T, cfg.n_steps, z)
                dW = increments_from_W(W)
            else:
                dt = cfg.T / cfg.n_steps
                dW = math.sqrt(dt) * z

            S_path = gbm_path_from_increments(cfg.S0, cfg.r, cfg.sigma, cfg.T, dW)

            if cfg.name == "asian_gbm":
                return disc * asian_arithmetic_call_payoff(S_path, cfg.K)

            # barrier up-and-out call
            dt = cfg.T / cfg.n_steps
            if barrier_continuous_weighting:
                payoff = barrier_uo_call_continuous_weighted(S_path, cfg.K, cfg.barrier_B, cfg.sigma, dt)
                return disc * payoff

            # discrete monitoring only (ablation)
            if np.any(S_path >= cfg.barrier_B):
                return 0.0
            return disc * max(S_path[-1] - cfg.K, 0.0)

        if cfg.name == "basket_call_gbm":
            d = cfg.d_assets
            z = z[:d]
            L = basket_cholesky(d, cfg.rho)
            X = L @ z

            drift = (cfg.r - 0.5 * cfg.sigma * cfg.sigma) * cfg.T
            vol = cfg.sigma * math.sqrt(cfg.T)
            S_T = cfg.S0 * np.exp(drift + vol * X)
            return disc * basket_call_payoff(S_T, cfg.K)

        raise ValueError(f"Unknown case name: {cfg.name}")

    if not antithetic:
        return one_payoff(Z)
    return 0.5 * (one_payoff(Z) + one_payoff(-Z))


def run_method(
    cfg: CaseConfig,
    method: str,
    N: int,
    R: int,
    use_brownian_bridge: bool,
    antithetic: bool,
    barrier_continuous_weighting: bool,
) -> Dict:
    # Runs R replications; each replication produces an estimator using N samples.
    t0 = time.time()

    if cfg.name in ("asian_gbm", "barrier_uo_gbm"):
        d = cfg.n_steps
    elif cfg.name == "basket_call_gbm":
        d = cfg.d_assets
    else:
        raise ValueError("Unsupported case")

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
            payoffs[i] = simulate_case_once(
                cfg,
                Z[i],
                use_brownian_bridge=use_brownian_bridge,
                antithetic=antithetic,
                barrier_continuous_weighting=barrier_continuous_weighting,
            )
        rep_est[rep] = float(np.mean(payoffs))

    elapsed = time.time() - t0

    tag = method
    if use_brownian_bridge:
        tag += "+BB"
    if antithetic:
        tag += "+AV"
    if cfg.name == "barrier_uo_gbm" and barrier_continuous_weighting:
        tag += "+Cont"

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
            "n_steps": cfg.n_steps,
            "barrier_continuous_weighting": bool(barrier_continuous_weighting),
        }
    }


def main():
    Ns = [2**k for k in range(8, 15)]  # 256 .. 16384
    R = 32

    cases = [
        CaseConfig(name="asian_gbm"),
        CaseConfig(name="barrier_uo_gbm", barrier_B=130.0),
        CaseConfig(name="basket_call_gbm", d_assets=16, rho=0.3),
    ]

    results = []
    for cfg in cases:
        for N in Ns:
            # MC baselines
            results.append(run_method(cfg, "MC", N, R, use_brownian_bridge=False, antithetic=False, barrier_continuous_weighting=True))
            results.append(run_method(cfg, "MC", N, R, use_brownian_bridge=False, antithetic=True,  barrier_continuous_weighting=True))

            # RQMC baselines
            if cfg.name in ("asian_gbm", "barrier_uo_gbm"):
                results.append(run_method(cfg, "RQMC_Sobol", N, R, use_brownian_bridge=False, antithetic=False, barrier_continuous_weighting=True))
                results.append(run_method(cfg, "RQMC_Sobol", N, R, use_brownian_bridge=True,  antithetic=False, barrier_continuous_weighting=True))
            else:
                results.append(run_method(cfg, "RQMC_Sobol", N, R, use_brownian_bridge=False, antithetic=False, barrier_continuous_weighting=False))

            print(f"[done] {cfg.name} N={N}")

    out = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Ns": Ns,
            "R": R,
            "notes": "Barrier uses continuous-monitoring correction via BB survival weights (no Bernoulli knockout). Basket uses equicorrelation Cholesky."
        },
        "results": results
    }

    with open("bench_results_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote bench_results_v2.json")


if __name__ == "__main__":
    main()
