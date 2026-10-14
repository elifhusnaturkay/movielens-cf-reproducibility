"""Run eval_protocols.py for every dataset, model and protocol, then print tables.

    python3 run_protocols.py                  # all missing runs
    python3 run_protocols.py ml100k svd       # only these
    python3 run_protocols.py --table --md     # tables only, as markdown

Runs one at a time (a 20M run needs several GB). Finished runs are skipped.
"""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(HERE, "..", "results", "eval.jsonl")
DATASETS = ["ml100k", "ml20m", "mlsmall"]
MODELS = ["svd", "own_svd", "cc", "own_cc", "dt", "dt20", "knn"]
PROTOCOLS = ([("random", None), ("temporal", None)] + [("kfold", f) for f in range(5)]
             + [("loo_last", None), ("loo_random", None)])
# The paper's rows. SVD on 20M is Table 4 p. 18; the tree only appears in
# Table 1 p. 13, which is the paper's Dataset 1 (ml-latest-small), not 20M.
PAPER = {
    ("mlsmall", "svd"): ("Table 1 (Dataset 1)", 0.87, 0.67, 0.78),
    ("mlsmall", "dt"): ("Table 1 (Dataset 1)", 1.38, 1.04, 0.74),
    ("ml20m", "svd"): ("Table 4 (20M)", 0.91, 0.90, 0.43),
    ("ml20m", "dt"): ("Table 1 (Dataset 1!)", 1.38, 1.04, 0.74),
    ("ml100k", "svd"): ("Table 1 (Dataset 1!)", 0.87, 0.67, 0.78),
    ("ml100k", "dt"): ("Table 1 (Dataset 1!)", 1.38, 1.04, 0.74),
}
NAMES = {"random": "random 80/20", "temporal": "temporal 80/20 (paper p. 6)",
         "kfold": "5-fold CV", "loo_last": "leave-last-one-out",
         "loo_random": "leave-random-one-out"}


def done():
    if not os.path.exists(JSONL):
        return {}
    rows = [json.loads(line) for line in open(JSONL)]
    return {(r["dataset"], r["model"], r["protocol"], r["fold"]): r for r in rows}


def summary(dataset, model, rows):
    """One line per protocol; the 5 folds collapse to mean ± std."""
    out = []
    for protocol in ["random", "temporal", "kfold", "loo_last", "loo_random"]:
        if protocol == "kfold":
            fs = [rows[(dataset, model, "kfold", f)] for f in range(5)
                  if (dataset, model, "kfold", f) in rows]
            if len(fs) < 5:
                continue
            m = {k: np.array([f[k] for f in fs]) for k in ("rmse", "mae", "r2")}
            out.append((NAMES[protocol],
                        *(f"{m[k].mean():.4f} ± {m[k].std():.4f}" for k in ("rmse", "mae", "r2")),
                        fs[0]["n_test"], "5 folds"))
        elif (dataset, model, protocol, None) in rows:
            r = rows[(dataset, model, protocol, None)]
            out.append((NAMES[protocol], f"{r['rmse']:.4f}", f"{r['mae']:.4f}",
                        f"{r['r2']:.4f}", r["n_test"], r["fingerprint"]))
    return out


def table(md=False):
    rows = done()
    for dataset in DATASETS:
        for model in MODELS:
            lines = summary(dataset, model, rows)
            if not lines:
                continue
            paper = PAPER.get((dataset, model))
            if md:
                print(f"\n**{dataset} — {model}**\n")
                print("| protocol | RMSE | MAE | R² | n_test | fingerprint |")
                print("|---|---|---|---|---|---|")
                if paper:
                    print(f"| *paper, {paper[0]}* | *{paper[1]}* | *{paper[2]}* | *{paper[3]}* | | |")
                for l in lines:
                    print("| " + " | ".join(str(x) for x in l) + " |")
            else:
                print(f"\n== {dataset}  {model}")
                print(f"{'protocol':<30}{'RMSE':>18}{'MAE':>18}{'R2':>18}{'n_test':>10}  fingerprint")
                if paper:
                    print(f"{'paper, ' + paper[0]:<30}{paper[1]:>18}{paper[2]:>18}{paper[3]:>18}")
                for l in lines:
                    print(f"{l[0]:<30}{l[1]:>18}{l[2]:>18}{l[3]:>18}{l[4]:>10}  {l[5]}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--table" not in args:
        datasets = [a for a in args if a in DATASETS] or DATASETS
        models = [a for a in args if a in MODELS] or MODELS
        have = done()
        for dataset in datasets:
            for model in models:
                for protocol, fold in PROTOCOLS:
                    if (dataset, model, protocol, fold) in have:
                        continue
                    cmd = [sys.executable, os.path.join(HERE, "eval_protocols.py"),
                           dataset, model, protocol] + ([str(fold)] if fold is not None else [])
                    print(">>", " ".join(cmd[2:]), flush=True)
                    subprocess.run(cmd, check=True, cwd=HERE)
    table(md="--md" in args)
