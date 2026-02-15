import json
import os
import argparse
import matplotlib.pyplot as plt


def load_results(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data["meta"], data["results"]


def choose_reference(results, case: str):
    # Reference for abs-error plot:
    # prefer RQMC_Sobol+BB at max N, else RQMC_Sobol, else MC+AV.
    case_rows = [r for r in results if r["case"] == case]
    maxN = max(r["N"] for r in case_rows)

    def pick(prefix: str):
        rows = [r for r in case_rows if r["N"] == maxN and r["method"].startswith(prefix)]
        if not rows:
            return None
        rows.sort(key=lambda x: x.get("estimate_std", float("inf")))
        return rows[0]["estimate_mean"]

    ref = pick("RQMC_Sobol+BB")
    if ref is None:
        ref = pick("RQMC_Sobol")
    if ref is None:
        ref = pick("MC+AV")
    if ref is None:
        ref = [r for r in case_rows if r["N"] == maxN][0]["estimate_mean"]
    return ref, maxN


def group_by_method(results, case: str):
    out = {}
    for r in results:
        if r["case"] != case:
            continue
        out.setdefault(r["method"], []).append(r)
    for m in out:
        out[m].sort(key=lambda x: x["N"])
    return out


def plot_case(meta, results, case: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    groups = group_by_method(results, case)
    ref, refN = choose_reference(results, case)

    # Std vs N
    plt.figure()
    for method, rows in groups.items():
        Ns = [r["N"] for r in rows]
        stds = [r.get("estimate_std", 0.0) for r in rows]
        plt.plot(Ns, stds, marker="o", label=method)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("N (paths per replication)")
    plt.ylabel("Std across R replications")
    plt.title(f"{case}: estimator variability (R={meta['R']})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{case}_std_vs_N.png"), dpi=200)
    plt.close()

    # Abs error vs cost (proxy, relative to ref)
    plt.figure()
    for method, rows in groups.items():
        cost = [r.get("cost_payoff_evals", r["N"] * r["R"]) for r in rows]
        err = [abs(r["estimate_mean"] - ref) for r in rows]
        plt.plot(cost, err, marker="o", label=method)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Cost (payoff evals = N * R)")
    plt.ylabel(f"|estimate - ref| (ref = best @ N={refN})")
    plt.title(f"{case}: abs error vs cost (proxy reference)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{case}_abs_err_vs_cost.png"), dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to benchmark JSON")
    ap.add_argument("--outdir", default="results/figures", help="Directory for figures")
    args = ap.parse_args()

    meta, results = load_results(args.input)
    cases = sorted(set(r["case"] for r in results))
    for c in cases:
        plot_case(meta, results, c, args.outdir)
    print(f"Wrote figures to {args.outdir}")


if __name__ == "__main__":
    main()
