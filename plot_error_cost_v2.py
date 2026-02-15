import json
import os
import argparse
import matplotlib.pyplot as plt


def load_results(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("meta", {}), data["results"]


def load_reference(ref_path: str):
    with open(ref_path, "r") as f:
        ref = json.load(f)
    # expected schema from compute_reference.py
    val = float(ref["reference"]["value"])
    meta = ref.get("meta", {})
    return val, meta


def choose_reference_proxy(results, case: str):
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
    return float(ref), maxN


def group_by_method(results, case: str):
    out = {}
    for r in results:
        if r["case"] != case:
            continue
        out.setdefault(r["method"], []).append(r)
    for m in out:
        out[m].sort(key=lambda x: x["N"])
    return out


def plot_case(meta, results, case: str, outdir: str, ref_val=None, ref_label="ref"):
    os.makedirs(outdir, exist_ok=True)
    groups = group_by_method(results, case)

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
    plt.title(f"{case}: estimator variability (R={meta.get('R','?')})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{case}_std_vs_N.png"), dpi=200)
    plt.close()

    # Abs error vs cost
    plt.figure()
    for method, rows in groups.items():
        cost = [r.get("cost_payoff_evals", r["N"] * r["R"]) for r in rows]
        err = [abs(r["estimate_mean"] - ref_val) for r in rows]
        plt.plot(cost, err, marker="o", label=method)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Cost (payoff evals = N * R)")
    plt.ylabel(f"|estimate - {ref_label}|")
    plt.title(f"{case}: abs error vs cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{case}_abs_err_vs_cost.png"), dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to benchmark JSON")
    ap.add_argument("--outdir", default="results/figures", help="Directory for figures")
    ap.add_argument("--reference", default=None, help="Optional reference JSON from compute_reference.py")
    args = ap.parse_args()

    meta, results = load_results(args.input)
    cases = sorted(set(r["case"] for r in results))

    ref_val = None
    ref_label = "ref"

    if args.reference:
        ref_val, ref_meta = load_reference(args.reference)
        ref_label = f"ref (N={ref_meta.get('N','?')},R={ref_meta.get('R','?')})"
    else:
        # per-case proxy reference
        pass

    for c in cases:
        if args.reference:
            plot_case(meta, results, c, args.outdir, ref_val=ref_val, ref_label=ref_label)
        else:
            proxy, refN = choose_reference_proxy(results, c)
            plot_case(meta, results, c, args.outdir, ref_val=proxy, ref_label=f"proxy@N={refN}")

    print(f"Wrote figures to {args.outdir}")


if __name__ == "__main__":
    main()
