"""Figure and tables for the fast SVR versions.

    python3 plot_svr_speed.py

Reads ../results/svr_speed.jsonl (all splits, RMSE) and svr_timing.jsonl (clean
timing runs, if there). Writes results/figures/svr_speed.png and prints the
markdown tables.
"""

import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"svr_rbf": "#eb6834", "linear_svr": "#52514e", "rbfs": "#2a78d6", "nys": "#1baf7a"}
SIZES = [100, 300, 1000, 3000]
MODELS = ["svr_rbf", "linear_svr"] + [f"rbfs_{n}" for n in SIZES] + [f"nys_{n}" for n in SIZES]
NAMES = {"svr_rbf": "RBF SVR (usual)", "linear_svr": "Linear SVR"}
NAMES.update({f"rbfs_{n}": f"RBF Sampler {n} + Linear SVR" for n in SIZES})
NAMES.update({f"nys_{n}": f"Nystroem {n} + Linear SVR" for n in SIZES})
TITLES = {"mlsmall": "MovieLens latest small", "ml100k": "MovieLens 100k"}


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path)]


def table(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["protocol"], r["model"])].append(r)
    return by


def kind(m):
    return m.split("_")[0] if m.startswith(("rbfs", "nys")) else m


def main():
    rmse = table(load("svr_speed.jsonl"))
    timing = table(load("svr_timing.jsonl"))
    # clean timing if there, else the times of the 5-fold runs
    tsrc, tprot = (timing, "random") if timing else (rmse, "kfold")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=160)
    for ax, ds in zip(axes, TITLES):
        for k in ("rbfs", "nys"):
            ms = [f"{k}_{n}" for n in SIZES]
            x = [np.median([r["train_s"] for r in tsrc[(ds, tprot, m)]]) for m in ms]
            y = [np.mean([r["rmse"] for r in rmse[(ds, "kfold", m)]]) for m in ms]
            ax.plot(x, y, "-o", color=COLOR[k], markersize=5,
                    label="RBF Sampler + Linear SVR" if k == "rbfs" else "Nystroem + Linear SVR")
            for xi, yi, n in zip(x, y, SIZES):
                ax.annotate(str(n), (xi, yi), xytext=(4, 4), textcoords="offset points",
                            fontsize=6.5, color=COLOR[k])
        for m in ("svr_rbf", "linear_svr"):
            x = np.median([r["train_s"] for r in tsrc[(ds, tprot, m)]])
            y = np.mean([r["rmse"] for r in rmse[(ds, "kfold", m)]])
            ax.plot(x, y, "s", color=COLOR[m], markersize=7, label=NAMES[m])
        ax.set_xscale("log")
        ax.set_title(TITLES[ds] + ", 5-fold CV", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("training time in seconds (log scale)", color=MUTED, fontsize=8)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=7.5)
    axes[0].set_ylabel("test RMSE (lower is better)", color=MUTED, fontsize=8)
    axes[1].legend(fontsize=7, frameon=False)
    fig.suptitle("Accuracy against training time (numbers = components)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "svr_speed.png"))

    for ds in TITLES:
        print(f"\n{TITLES[ds]}\n| Model | random 80/20 | temporal 80/20 | 5-fold CV | "
              f"training time (s) | prediction time (s) |")
        print("|---|---|---|---|---|---|")
        for m in MODELS:
            cells = []
            for p in ("random", "temporal"):
                cells.append(f"{rmse[(ds, p, m)][0]['rmse']:.4f}")
            v = [r["rmse"] for r in rmse[(ds, "kfold", m)]]
            cells.append(f"{np.mean(v):.4f} ± {np.std(v):.4f}")
            tr = np.median([r["train_s"] for r in tsrc[(ds, tprot, m)]])
            pr = np.median([r["predict_s"] for r in tsrc[(ds, tprot, m)]])
            print(f"| {NAMES[m]} | " + " | ".join(cells) + f" | {tr:.1f} | {pr:.2f} |")


if __name__ == "__main__":
    main()
