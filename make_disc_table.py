"""
Convert discretization_sweep_heston_asian.json into a LaTeX table.

Usage:
  python make_disc_table.py --input disc_sweep_heston_asian.json --out disc_table.tex
"""
import json
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="disc_table.tex")
    args = ap.parse_args()

    with open(args.input,"r") as f:
        data=json.load(f)

    rows=data["rows"]
    meta=data["meta"]

    lines=[]
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Heston Asian call discretization sweep. Fixed sampling budget $(N,R)=(%d,%d)$; scheme=%s.}"
                 % (meta["N"], meta["R"], meta["scheme"]))
    lines.append(r"\label{tab:heston_disc_sweep}")
    lines.append(r"\begin{tabular}{r r r}")
    lines.append(r"\toprule")
    lines.append(r"$n$ (steps) & Estimate & Std across replications \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(r"%d & %.8f & %.8f \\" % (r["n_steps"], r["estimate_mean"], r["estimate_std"]))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(args.out,"w") as f:
        f.write("\n".join(lines))
    print("Wrote", args.out)

if __name__=="__main__":
    main()
