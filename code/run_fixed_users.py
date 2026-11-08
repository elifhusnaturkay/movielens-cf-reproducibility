"""Fix the user side of co-clustering, change the movie side.

    python3 run_fixed_users.py              # runs what is missing, then draws
    python3 run_fixed_users.py plot         # only draws

viz.py found the best number of user groups at K = 2 or 3. So the user side is
fixed at K = 2 and K = 3 and the movie side takes L = 1, 2, 3, 5, 7, 10, 15, 20.
Six bases, 5-fold CV, latest-small and ml-100k.

Most of these cells are already in ../results/grid.jsonl (the K x L grid, same
code, same folds). Only L = 7 is new. New rows go to the same file.

Output: ../results/figures/fixed_users_<ds>.png and ../results/fixed_users.md
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_bases as rb
from bases import BASES

KS = [2, 3]
LS = [1, 2, 3, 5, 7, 10, 15, 20]
DS = ["mlsmall", "ml100k"]
NAMES = {"mlsmall": "Latest small", "ml100k": "100k"}
import tema  # noqa: E402

tema.apply()
PATH = os.path.join(rb.OUT, "grid.jsonl")


def run_missing():
    have = rb.done(PATH, ["dataset", "basis", "protocol", "fold", "k", "l"])
    for dataset in DS:
        ds = rb.DATASETS[dataset]
        raw = ds["load"]()
        shape = (int(raw[:, 0].max()) + 1, int(raw[:, 1].max()) + 1)
        for fold in range(5):
            todo = [(b, k, l) for b in BASES for k in KS for l in LS
                    if (dataset, b, "kfold", fold, k, l) not in have]
            if not todo:
                continue
            train, test = rb.get_split(raw, "kfold", fold)
            fp = rb.cs.fingerprint(test)
            print(f"[split] {dataset} kfold fold={fold} fingerprint={fp}  {len(todo)} to run", flush=True)
            for b, k, l in todo:
                row = dict(dataset=dataset, protocol="kfold", fold=fold, fingerprint=fp,
                           **rb.run_one(b, k, l, train, test, shape, ds["scale"]), note="fixed users")
                print(f"  {b} {k} x {l:<2} RMSE={row['rmse']:.4f}  {row['fit_s']}s", flush=True)
                with open(PATH, "a") as f:
                    f.write(json.dumps(row) + "\n")


def table():
    """{(ds, basis, k, l): [rmse of the 5 folds]}, only 5-fold rows"""
    m = defaultdict(list)
    for line in open(PATH):
        r = json.loads(line)
        if r["protocol"] == "kfold":
            m[(r["dataset"], r["basis"], r["k"], r["l"])].append(r["rmse"])
    return m


def best_overall(m, ds, b):
    return min((np.mean(v), k, l) for (d, bb, k, l), v in m.items() if d == ds and bb == b and len(v) == 5)


def plot(m):
    lines = ["# User side fixed, movie side changed (5-fold mean RMSE)", ""]
    for ds in DS:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharey=True)
        fig.subplots_adjust(hspace=0.35)
        x = np.arange(len(LS)); w = 0.13
        for ax, k in zip(axes, KS):
            for j, b in enumerate(BASES):
                mean = [np.mean(m[(ds, b, k, l)]) for l in LS]
                std = [np.std(m[(ds, b, k, l)]) for l in LS]
                ax.bar(x + (j - 2.5) * w, mean, w, yerr=std, color=tema.BASIS[b], edgecolor=tema.EDGE[b],
                       linewidth=0.7, label=tema.BASIS_NAME[b], error_kw=dict(lw=0.7, capsize=1.5, ecolor=tema.INK))
            ax.set_xticks(x, [f"{k} × {l}" for l in LS])
            ax.set_title(f"user groups fixed at K = {k}, movie groups L changes", fontsize=10.5)
            ax.set_ylabel("RMSE (5-fold mean)"); tema.style(ax)
        lo = min(np.mean(m[(ds, b, k, l)]) for b in BASES for k in KS for l in LS)
        hi = max(np.mean(m[(ds, b, k, l)]) for b in BASES for k in KS for l in LS)
        axes[0].set_ylim(lo - 0.02, min(hi + 0.02, lo + 0.25))
        axes[0].legend(ncol=3, fontsize=8, loc="upper left")
        fig.suptitle(f"{NAMES[ds]}: co-clustering with the user side fixed (lower is better, bar = ± std over folds)",
                     fontsize=11.5)
        fig.savefig(os.path.join(rb.OUT, "figures", f"fixed_users_{ds}.png"), dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", f"fixed_users_{ds}.png")

        lines += [f"## {NAMES[ds]}", "", "| Basis | " + " | ".join(f"K={k} best L" for k in KS) +
                  " | Best of the whole grid |", "|---|" + "---|" * (len(KS) + 1)]
        for b in BASES:
            cells = []
            for k in KS:
                v, l = min((np.mean(m[(ds, b, k, l)]), l) for l in LS)
                cells.append(f"{v:.4f} ({k} × {l})")
            bv, bk, bl = best_overall(m, ds, b)
            lines.append(f"| {b} | " + " | ".join(cells) + f" | {bv:.4f} ({bk} × {bl}) |")
        lines += ["", "| Basis | " + " | ".join(f"{k} × {l}" for k in KS for l in LS) + " |",
                  "|---|" + "---|" * (len(KS) * len(LS))]
        for b in BASES:
            lines.append(f"| {b} | " + " | ".join(f"{np.mean(m[(ds, b, k, l)]):.4f}" for k in KS for l in LS) + " |")
        lines.append("")
    with open(os.path.join(rb.OUT, "fixed_users.md"), "w") as f:
        f.write("\n".join(lines))
    print("wrote fixed_users.md")


if __name__ == "__main__":
    if "plot" not in sys.argv:
        run_missing()
    plot(table())
