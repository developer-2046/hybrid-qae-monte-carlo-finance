#!/usr/bin/env python3
"""
GPU benchmark for Heston Asian (arithmetic) option pricing.

What this solves:
- Your CPU benchmark script is loop-heavy and can be slow at R=32 over many N's.
- This version is vectorized over paths using PyTorch (works on Colab A100/H100).
- Includes MC, MC+AV (antithetic), and RQMC (scrambled Sobol) baselines.
- Uses the Andersen QE variance scheme (recommended for Heston).

Outputs a JSON compatible with the existing plotting scripts (results list with replications).

Example (Colab / CUDA):
  python bench_heston_asian_torch.py --scheme qe --n_steps 256 --R 32 --N_max_pow 14 --batch 65536 --device cuda --out bench_results_heston_asian_qe_torch.json

Notes:
- Brownian-bridge (BB) is a Brownian-path construction; it doesn't naturally pair with QE variance sampling.
  So we benchmark RQMC(Sobol) without BB for the QE scheme.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Any

import torch


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
    K: float = 100.0


def inv_norm(u: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(u.dtype).eps
    u = torch.clamp(u, eps, 1 - eps)
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


def qe_variance_step(v: torch.Tensor, u: torch.Tensor, z: torch.Tensor, p: HestonParams, dt: float, psi_c: float = 1.5) -> torch.Tensor:
    v = torch.clamp(v, min=0.0)
    kappa, theta, xi = float(p.kappa), float(p.theta), float(p.xi)

    exp_kdt = math.exp(-kappa * dt)
    m = theta + (v - theta) * exp_kdt

    s2 = (v * (xi * xi) * exp_kdt * (1 - exp_kdt) / kappa) + (theta * (xi * xi) * (1 - exp_kdt) ** 2 / (2 * kappa))
    s2 = torch.clamp(s2, min=0.0)
    m_safe = torch.clamp(m, min=1e-12)
    psi = s2 / (m_safe * m_safe)

    mask_qg = psi <= psi_c
    mask_exp = ~mask_qg

    v_next = torch.zeros_like(v)

    if mask_qg.any():
        psi_q = psi[mask_qg]
        m_q = m_safe[mask_qg]
        z_q = z[mask_qg]

        invpsi = 1.0 / torch.clamp(psi_q, min=1e-12)
        term = torch.sqrt(torch.clamp(2.0 * invpsi, min=0.0))
        inner = torch.sqrt(torch.clamp(2.0 * invpsi - 1.0, min=0.0))
        b2 = 2.0 * invpsi - 1.0 + term * inner
        b2 = torch.clamp(b2, min=0.0)
        b = torch.sqrt(b2)
        a = m_q / (1.0 + b2)
        v_next[mask_qg] = a * (b + z_q) ** 2

    if mask_exp.any():
        psi_e = psi[mask_exp]
        m_e = m_safe[mask_exp]
        u_e = u[mask_exp]

        p0 = (psi_e - 1.0) / (psi_e + 1.0)
        p0 = torch.clamp(p0, 0.0, 1.0)
        beta = (1.0 - p0) / m_e
        beta = torch.clamp(beta, min=1e-12)

        one_minus_u = torch.clamp(1.0 - u_e, min=1e-12)
        one_minus_p0 = torch.clamp(1.0 - p0, min=1e-12)
        v_e = -torch.log(one_minus_u / one_minus_p0) / beta
        v_e = torch.where(u_e <= p0, torch.zeros_like(v_e), v_e)
        v_next[mask_exp] = v_e

    return torch.clamp(v_next, min=0.0)


def heston_asian_qe_from_uniforms(u: torch.Tensor, p: HestonParams) -> torch.Tensor:
    """
    u: (B, 2*n_steps) uniforms.
      first n: variance driver (u_var)
      second n: independent Gaussian driver for log-price (u_ind -> z_ind)
    returns: (B,) discounted payoff
    """
    n = p.n_steps
    dt = p.T / n

    u_var = u[:, :n]
    u_ind = u[:, n:2*n]
    z_var = inv_norm(u_var)
    z_ind = inv_norm(u_ind)

    v = torch.full((u.shape[0],), float(p.v0), device=u.device, dtype=u.dtype)
    v = torch.clamp(v, min=0.0)
    logS = torch.full((u.shape[0],), math.log(p.S0), device=u.device, dtype=u.dtype)
    sumS = torch.zeros_like(logS)

    rho = float(p.rho)
    xi = float(p.xi)
    kappa = float(p.kappa)
    theta = float(p.theta)

    for i in range(n):
        v_next = qe_variance_step(v, u_var[:, i], z_var[:, i], p, dt)
        I = 0.5 * (v + v_next) * dt
        I = torch.clamp(I, min=0.0)

        if xi > 0.0:
            corr_term = (rho / xi) * (v_next - v - kappa * theta * dt + kappa * I)
        else:
            corr_term = 0.0

        diff = torch.sqrt(torch.clamp((1.0 - rho * rho) * I, min=0.0)) * z_ind[:, i]

        logS = logS + p.r * dt - 0.5 * I + corr_term + diff
        S = torch.exp(logS)
        sumS = sumS + S
        v = v_next

    A = sumS / float(n)
    payoff = torch.clamp(A - float(p.K), min=0.0)
    disc = math.exp(-p.r * p.T)
    return payoff * disc


def draw_sobol_uniforms(d: int, n: int, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
    u = eng.draw(n).to(device=device, dtype=dtype)
    eps = torch.finfo(dtype).eps
    return torch.clamp(u, eps, 1 - eps)


def run_replication(
    p: HestonParams,
    method: str,
    N: int,
    seed: int,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """
    Returns one replication mean (mean discounted payoff over N samples).
    Methods:
      - "MC"
      - "MC+AV"
      - "RQMC_Sobol"
    """
    d = 2 * p.n_steps
    total = 0.0
    count = 0


    if method == "MC":
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        remaining = N
        while remaining > 0:
            b = batch if remaining >= batch else remaining
            # generate uniforms for a shared pipeline (keeps payoff function identical)
            u = torch.rand((b, d), generator=gen, device=device, dtype=dtype)
            eps = torch.finfo(dtype).eps
            u = torch.clamp(u, eps, 1 - eps)
            pay = heston_asian_qe_from_uniforms(u, p)
            total += float(pay.sum().item())
            count += b
            remaining -= b
        return total / count

    if method == "MC+AV":
        # antithetic on Gaussian space: U -> Z -> -Z -> U'
        # easiest: sample Z directly, then compute U=Phi(Z) and U'=Phi(-Z)=1-U.
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        remaining = N
        while remaining > 0:
            b = batch if remaining >= batch else remaining
            z = torch.randn((b, d), generator=gen, device=device, dtype=dtype)
            # map to uniforms via Phi(z)=0.5*(1+erf(z/sqrt(2)))
            u = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
            u_anti = 1.0 - u  # Phi(-z)
            eps = torch.finfo(dtype).eps
            u = torch.clamp(u, eps, 1 - eps)
            u_anti = torch.clamp(u_anti, eps, 1 - eps)
            pay = 0.5 * (heston_asian_qe_from_uniforms(u, p) + heston_asian_qe_from_uniforms(u_anti, p))
            total += float(pay.sum().item())
            count += b
            remaining -= b
        return total / count

    if method == "RQMC_Sobol":
        # Draw Sobol uniforms in blocks (preserves low-discrepancy)
        eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
        remaining = N
        while remaining > 0:
            b = batch if remaining >= batch else remaining
            u = eng.draw(b).to(device=device, dtype=dtype)
            eps = torch.finfo(dtype).eps
            u = torch.clamp(u, eps, 1 - eps)
            pay = heston_asian_qe_from_uniforms(u, p)
            total += float(pay.sum().item())
            count += b
            remaining -= b
        return total / count

    raise ValueError(f"Unknown method: {method}")


def bench(
    p: HestonParams,
    Ns: List[int],
    R: int,
    batch: int,
    seed0: int,
    device: str,
    dtype_str: str,
    out_path: str,
) -> None:
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    dtype = torch.float32 if dtype_str == "float32" else torch.float64

    methods = ["MC", "MC+AV", "RQMC_Sobol"]
    results: List[Dict[str, Any]] = []

    for N in Ns:
        for method in methods:
            t0 = time.time()
            rep_means: List[float] = []
            for rep in range(R):
                seed = seed0 + 1000 * rep + 17 + (hash(method) % 997)
                m = run_replication(p, method, N, seed, batch, dev, dtype)
                rep_means.append(float(m))
            elapsed = time.time() - t0

            mean = float(sum(rep_means) / len(rep_means))
            if R > 1:
                m0 = mean
                var = sum((x - m0) ** 2 for x in rep_means) / (R - 1)
                std = float(math.sqrt(var))
            else:
                std = 0.0

            # cost metric: payoff evaluations (AV uses 2 payoffs per sample)
            payoff_evals = N * R * (2 if method == "MC+AV" else 1)

            results.append({
                "case": "asian_heston",
                "method": method,
                "scheme": "qe_torch",
                "N": int(N),
                "R": int(R),
                "estimate_mean": mean,
                "estimate_std": std,
                "replications": rep_means,
                "cost_payoff_evals": int(payoff_evals),
                "wall_time_sec": float(elapsed),
                "notes": {
                    "n_steps": int(p.n_steps),
                    "device": str(dev),
                    "dtype": dtype_str,
                    "batch": int(batch),
                }
            })
            print(f"[done] N={N} method={method} mean={mean:.8f} std={std:.6f} time={elapsed:.2f}s", flush=True)

    out = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "case": "asian_heston",
            "scheme": "qe_torch",
            "device": str(dev),
            "dtype": dtype_str,
            "Ns": Ns,
            "R": R,
            "batch": batch,
            "params": p.__dict__,
        },
        "results": results
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_results_heston_asian_qe_torch.json")
    ap.add_argument("--seed0", type=int, default=12345)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["float32","float64"], default="float32")
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--N_min_pow", type=int, default=8)
    ap.add_argument("--N_max_pow", type=int, default=14)

    # model
    ap.add_argument("--S0", type=float, default=100.0)
    ap.add_argument("--v0", type=float, default=0.04)
    ap.add_argument("--r", type=float, default=0.03)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--theta", type=float, default=0.04)
    ap.add_argument("--xi", type=float, default=0.5)
    ap.add_argument("--rho", type=float, default=-0.7)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--n_steps", type=int, default=256)
    ap.add_argument("--K", type=float, default=100.0)

    args = ap.parse_args()
    p = HestonParams(
        S0=args.S0, v0=args.v0, r=args.r,
        kappa=args.kappa, theta=args.theta, xi=args.xi, rho=args.rho,
        T=args.T, n_steps=args.n_steps, K=args.K
    )
    Ns = [2**k for k in range(args.N_min_pow, args.N_max_pow + 1)]
    bench(p, Ns, args.R, args.batch, args.seed0, args.device, args.dtype, args.out)


if __name__ == "__main__":
    main()
