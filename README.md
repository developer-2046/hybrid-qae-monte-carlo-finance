# QMC vs QAE for Option Pricing (Full-Stack Benchmarks)

End-to-end benchmarking of **best-practice classical (R)QMC** against **quantum amplitude estimation (QAE/IQAE)** pipelines for option pricing problems where **no closed form** is available (path-dependent, stochastic volatility, multi-asset).

This repo is designed to be:
- **Reproducible**: deterministic configs, saved outputs, plot scripts.
- **Fair**: compares against strong classical baselines (scrambled Sobol + dimension reduction), not strawmen.
- **Full-stack**: includes discretization, payoff normalization, and an explicit error/cost accounting.

## Why this exists

Most “quantum option pricing” demos compare against **plain Monte Carlo** or closed-form Black–Scholes toy setups. In practice, the real classical competitor is **randomized quasi–Monte Carlo (RQMC)** using **scrambled Sobol** points plus **Brownian bridge / PCA** style dimension reduction for Brownian paths.

This project focuses on the realistic question:

> When you benchmark against best-practice (R)QMC, where does a hybrid QAE/IQAE stack still have a compelling advantage (and where does it not)?

## Problems covered

Benchmark suite targets payoffs and models where Monte Carlo is standard:

- **Asian options** (path-dependent)
- **Barrier options** (discontinuous payoffs; barrier crossing bias)
- **Basket options** (high-dimensional)
- Extensions planned: **Heston stochastic volatility**, **VaR/CVaR**, and early-exercise approximations.

## Methods compared

### Classical baselines (best-practice)
- **MC** (crude Monte Carlo)
- **MC + variance reduction** (antithetic variates; control variates where applicable)
- **RQMC (scrambled Sobol)** with:
  - standard sequential Brownian construction
  - **Brownian bridge** path construction (dimension reduction)
  - optional PCA/LT constructions (roadmap)

### Quantum estimators (hybrid stack)
- **QAE / Iterative QAE (IQAE)** to estimate expectations via amplitude estimation
- Two implementation modes:
  - **Mode A (hardware-feasible)**: QROM/lookup over precomputed payoffs (small instances; noise studies)
  - **Mode B (scaling analysis)**: coherent path simulation + reversible payoff evaluation (resource accounting)

## What “full-stack” means here

We track and report the end-to-end error budget:

- **SDE discretization error** (time stepping bias)
- **truncation/normalization bias** (mapping payoff to [0,1] for amplitude estimation)
- **estimation error** (MC vs RQMC vs QAE/IQAE)
- **noise bias** (noisy simulation / hardware; optional mitigation)

This prevents “quadratic speedup” claims that ignore the expensive parts.

## Repository layout

```
.
├── paper/                 # LaTeX draft + bibliography (SISC-style)
├── bench/                 # Classical benchmarks (MC, VR, RQMC) + configs
├── quantum/               # QAE/IQAE implementations, simulators, resource accounting
├── results/               # Generated outputs (JSON/CSV) + figures
├── configs/               # Experiment configs (YAML/JSON)
└── scripts/               # Plotting and utility scripts
```

## Quickstart (classical benchmarks)

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run baseline suite
```bash
python bench/bench_gbm_asian_barrier_basket.py
```

Outputs:
- `bench_results.json` (raw replication outputs)
- plots are generated via scripts in `scripts/` (see below)

## Plotting

After running benchmarks:
```bash
python scripts/plot_error_cost.py --input bench_results.json --outdir results/figures
```

Produces:
- Error vs cost curves (MC, MC+VR, RQMC, RQMC+BB)
- Confidence intervals for RQMC via independent scrambles

## Reproducibility

- All experiments are driven by config files in `configs/`
- Outputs are saved to `results/` in machine-readable format
- Plots are generated from raw outputs (no manual editing)

## Roadmap

- [ ] Add Heston (stochastic volatility) benchmark suite
- [ ] Add PCA/LT path constructions for (R)QMC
- [ ] Implement IQAE pipeline and oracle cost accounting
- [ ] Noisy simulation + readout mitigation evaluation
- [ ] Small-scale hardware runs (optional, if access available)
- [ ] Paper figures + final SISC submission draft in `paper/`

## References (selected)

- Classical Monte Carlo in finance: Glasserman, *Monte Carlo Methods in Financial Engineering*
- (R)QMC in finance: scrambled Sobol + dimension reduction (Brownian bridge/PCA/LT)
- Quantum amplitude estimation and IQAE: foundational QAE + iterative variants
- Quantum option pricing: QAE-based option pricing frameworks

(Full BibTeX and citations live in `paper/refs.bib`.)

## License
MIT License (or update as needed).
