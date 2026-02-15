import math
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional

import numpy as np
from scipy.stats import norm
from scipy.stats import qmc


# ----------------------------
# Models / Payoffs
# ----------------------------
def asian_arithmetic_call_payoff(S_path: np.ndarray, K: float) -> float:
    # S_path shape: (n_steps+1,)
    A = np.mean(S_path[1:])  # average over monitoring times (exclude t=0)
    return max(A - K, 0.0)


def barrier_up_and_out_call_payoff(S_path: np.ndarray, K: float, B: float) -> float:
    # Discrete monitoring (we'll optionally add BB crossing correction separately)
    if np.any(S_path >= B):
        return 0.0
    return max(S_path[-1] - K, 0.0)


def basket_call_payoff(S_T: np.ndarray, K: float, weights: Optional[np.ndarray] = None) -> float:
    # S_T shape: (d_assets,)
    if weights is None:
        weights = np.ones_like(S_T) / len(S_T)
    basket = float(np.dot(weights, S_T))
    return max(basket - K, 0.0)


# ----------------------------
# GBM simulation utilities
# ----------------------------
def gbm_path_from_increments(
    S0: float, r: float, sigma: float, T: float, dW: np.ndarray
) -> np.ndarray:
    """
    dW: shape (n_steps,), Brownian increments
    returns S_path: shape (n_steps+1,)
    """
    n_steps = dW.shape[0]
    dt = T / n_steps
    S = np.empty(n_steps + 1, dtype=float)
    S[0] = S0
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma
    logS = math.log(S0)
    for i in range(n_steps):
        logS = logS + drift + vol * dW[i]
        S[i + 1] = math.exp(logS)
    return S


# ----------------------------
# Brownian bridge construction (for dimension reduction)
# ----------------------------
def brownian_bridge_W(T: float, n_steps: int, z: np.ndarray) -> np.ndarray:
    """
    Construct Brownian motion values W(t_i) at i=0..n_steps using a simple
    midpoint-recursion Brownian bridge fill.
    Requires z of length n_steps (standard normals).
    NOTE: best behavior when n_steps is a power of 2.
    """
    assert z.shape[0] == n_steps, "z must have length n_steps"
    t = np.linspace(0.0, T, n_steps + 1)
    W = np.full(n_steps + 1, np.nan, dtype=float)
    W[0] = 0.0
    W[n_steps] = math.sqrt(T) * z[0]

    k = 1
    intervals = [(0, n_steps)]
    while intervals:
        left, right = intervals.pop(0)
        if right - left <= 1:
            continue
        mid = (left + right) // 2
        if not np.isnan(W[mid]):
            continue

        tl, tm, tr = t[left], t[mid], t[right]
        # Conditional Brownian bridge at tm given endpoints
        mean = ((tr - tm) / (tr - tl)) * W[left] + ((tm - tl) / (tr - tl)) * W[right]
        var = (tm - tl) * (tr - tm) / (tr - tl)
        W[mid] = mean + math.sqrt(max(var, 0.0)) * z[k]
        k += 1

        intervals.append((left, mid))
        intervals.append((mid, right))

    # Fill any remaining NaNs sequentially (shouldn't happen for power-of-2 steps, but safe)
    nan_idx = np.where(np.isnan(W))[0]
    for idx in nan_idx:
        # fallback: unconditional
        W[idx] = math.sqrt(t[idx]) * z[min(k, n_steps - 1)]
        k += 1

    return W


def increments_from_W(W: np.ndarray) -> np.ndarray:
    return np.diff(W)


# ----------------------------
# QMC / RQMC generators
# ----------------------------
def sobol_normals_base2(n: int, d: int, scramble: bool, seed: int) -> np.ndarray:
    """
    Generate n=2^m Sobol points in [0,1)^d (optionally scrambled),
    map to N(0,1) via inverse CDF.
    """
    m = int(round(math.log2(n)))
    if 2**m != n:
        raise ValueError("n must be a power of 2 for random_base2.")
    eng = qmc.Sobol(d=d, scramble=scramble, seed=seed)
    u = eng.random_base2(m=m)
    # avoid infs in norm.ppf
    eps = np.finfo(float).eps
    u = np.clip(u, eps, 1 - eps)
    return norm.ppf(u)


# ----------------------------
# Barrier crossing correction (GBM) via Brownian bridge in log-space
# ----------------------------
def up_and_out_cross_prob_log_bridge(x0: float, x1: float, logB: float, sigma: float, dt: float) -> float:
    """
    Given log endpoints x0, x1 below logB, crossing probability for a Brownian bridge.
    Drift cancels conditional on endpoints; variance is sigma^2 * dt in log space per step.
    """
    if x0 >= logB or x1 >= logB:
        return 1.0
    denom = (sigma * sigma) * dt
    if denom <= 0:
        return 0.0
    return math.exp(-2.0 * (logB - x0) * (logB - x1) / denom)


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
    n_steps: int = 256  # power of 2 helps QMC + BB
    barrier_B: float = 130.0
    d_assets: int = 16
    rho: float = 0.3  # equicorrelation for basket
    seed0: int = 12345


def simulate_case_once(
    cfg: CaseConfig,
    Z: np.ndarray,
    method: str,
    use_brownian_bridge: bool,
    antithetic: bool,
    barrier_bb_correction: bool,
) -> float:
    """
    Returns discounted payoff for one "path/sample" (or averaged antithetic pair).
    Z dims depend on case:
      - asian/barrier: n_steps normals
      - basket: d_assets normals
    """
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
            if not barrier_bb_correction:
                return disc * barrier_up_and_out_call_payoff(S_path, cfg.K, cfg.barrier_B)

            # Brownian-bridge crossing correction stepwise in log space:
            # We simulate the discrete path but probabilistically knock out within each step.
            dt = cfg.T / cfg.n_steps
            logB = math.log(cfg.barrier_B)
            logS = np.log(S_path)
            # if already crossed at discrete times, it's out
            if np.any(S_path >= cfg.barrier_B):
                return 0.0
            # otherwise sample crossing within each interval
            for i in range(cfg.n_steps):
                p = up_and_out_cross_prob_log_bridge(logS[i], logS[i+1], logB, cfg.sigma, dt)
                if p <= 0.0:
                    continue
                # use an extra RNG from z tail via a deterministic hash-like mapping
                u = 0.5 * (1.0 + math.erf(z[(i * 13) % cfg.n_steps] / math.sqrt(2.0)))
                if u < p:
                    return 0.0
            return disc * max(S_path[-1] - cfg.K, 0.0)

        elif cfg.name == "basket_call_gbm":
            # One-step correlated normals -> terminal prices
            d = cfg.d_assets
            z = z[:d]
            # equicorrelation matrix via one-factor model
            rho = cfg.rho
            g0 = z[0]
            eps = z[1:]
            # Build correlated normals:
            X = np.empty(d, dtype=float)
            X[0] = g0
            if d > 1:
                X[1:] = rho * g0 + math.sqrt(max(1.0 - rho * rho, 0.0)) * eps

            drift = (cfg.r - 0.5 * cfg.sigma * cfg.sigma) * cfg.T
            vol = cfg.sigma * math.sqrt(cfg.T)
            S_T = cfg.S0 * np.exp(drift + vol * X)
            return disc * basket_call_payoff(S_T, cfg.K)

        else:
            raise ValueError(f"Unknown case name: {cfg.name}")

    if not antithetic:
        return one_payoff(Z)

    # Antithetic variates: average payoff(z) and payoff(-z)
    return 0.5 * (one_payoff(Z) + one_payoff(-Z))


def run_method(
    cfg: CaseConfig,
    method: str,
    N: int,
    R: int,
    use_brownian_bridge: bool,
    antithetic: bool,
    barrier_bb_correction: bool,
) -> Dict:
    """
    Runs R replications; each replication produces an estimator using N samples.
    For MC: each replication uses a fresh RNG seed.
    For RQMC: each replication uses a fresh scramble seed.
    """
    t0 = time.time()

    # Dimension of the driving normals per sample
    if cfg.name in ("asian_gbm", "barrier_uo_gbm"):
        d = cfg.n_steps
    elif cfg.name == "basket_call_gbm":
        d = cfg.d_assets
    else:
        raise ValueError("unsupported case")

    rep_est = np.empty(R, dtype=float)

    for rep in range(R):
        seed = cfg.seed0 + 1000 * rep + 17
        if method == "MC":
            rng = np.random.default_rng(seed)
            Z = rng.standard_normal(size=(N, d))
        elif method == "RQMC_Sobol":
            # scrambled Sobol (SciPy): LMS+shift scrambling
            Z = sobol_normals_base2(N, d, scramble=True, seed=seed)
        else:
            raise ValueError(f"Unknown method: {method}")

        payoffs = np.empty(N, dtype=float)
        for i in range(N):
            payoffs[i] = simulate_case_once(
                cfg, Z[i], method, use_brownian_bridge, antithetic, barrier_bb_correction
            )
        rep_est[rep] = float(np.mean(payoffs))

    elapsed = time.time() - t0
    return {
        "case": cfg.name,
        "method": method + ("+BB" if use_brownian_bridge else "") + ("+AV" if antithetic else ""),
        "N": N,
        "R": R,
        "estimate_mean": float(np.mean(rep_est)),
        "estimate_std": float(np.std(rep_est, ddof=1)) if R > 1 else 0.0,
        "replications": rep_est.tolist(),
        "cost_payoff_evals": int(N * R),
        "wall_time_sec": float(elapsed),
        "notes": {
            "barrier_bb_correction": bool(barrier_bb_correction),
            "n_steps": cfg.n_steps,
        }
    }


def main():
    # Default benchmark grid: N must be power-of-2 for Sobol base2
    Ns = [2**k for k in range(8, 15)]  # 256 .. 16384
    R = 32  # number of replications/scrambles to get error bars

    cases = [
        CaseConfig(name="asian_gbm"),
        CaseConfig(name="barrier_uo_gbm", barrier_B=130.0),
        CaseConfig(name="basket_call_gbm", d_assets=16, rho=0.3),
    ]

    results = []

    for cfg in cases:
        for N in Ns:
            # MC baselines
            results.append(run_method(cfg, "MC", N, R, use_brownian_bridge=False, antithetic=False, barrier_bb_correction=True))
            results.append(run_method(cfg, "MC", N, R, use_brownian_bridge=False, antithetic=True, barrier_bb_correction=True))

            # RQMC baselines: standard + Brownian bridge for path-based cases
            if cfg.name in ("asian_gbm", "barrier_uo_gbm"):
                results.append(run_method(cfg, "RQMC_Sobol", N, R, use_brownian_bridge=False, antithetic=False, barrier_bb_correction=True))
                results.append(run_method(cfg, "RQMC_Sobol", N, R, use_brownian_bridge=True, antithetic=False, barrier_bb_correction=True))
            else:
                results.append(run_method(cfg, "RQMC_Sobol", N, R, use_brownian_bridge=False, antithetic=False, barrier_bb_correction=False))

            print(f"[done] {cfg.name} N={N}")

    out = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Ns": Ns,
            "R": R
        },
        "results": results
    }
    with open("bench_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote bench_results.json")


if __name__ == "__main__":
    main()
