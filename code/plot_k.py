"""Draw the K vs RMSE plots from results/k_sweep.jsonl and print the tables.

    python3 plot_k.py          # PNGs into results/, markdown tables on screen

One plot per dataset. Blue = truncated SVD, orange = Funk SVD (Surprise's SVD).
5-fold runs are drawn as mean with a band of +- 1 std; random 80/20 as a line.
The grey dashed line is K = 5, the paper's choice. The black diamond is the
paper's own SVD number for that dataset.
"""

import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
JSONL = os.path.join(OUT, "k_sweep.jsonl")

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"tsvd": "Truncated SVD", "funk": "Funk SVD (Surprise)"}
TITLES = {"mlsmall": "MovieLens latest small (paper Dataset 1)",
          "ml100k": "MovieLens 100k ",
          "ml20m": "MovieLens 20M (paper Dataset 2)"}
# paper: Table 1 p. 13 (Dataset 1, 5-fold), Table 4 p. 18 (Dataset 2)
PAPER = {"mlsmall": 0.8739, "ml20m": 0.91}   # p. 13 text: "0.8739 ... five-fold"


def load():
    rows = [json.loads(line) for line in open(JSONL)]
    runs = collections.defaultdict(list)
    sing = {}
    for r in rows:
        if r["model"] == "singular_values":
            sing[r["dataset"]] = np.array(r["s"])
        else:
            runs[(r["dataset"], r["model"], r["protocol"])].append(r)
    return runs, sing


def curve(rows):
    """K list, mean RMSE, std RMSE (std = 0 for a single run)."""
    by_k = collections.defaultdict(list)
    for r in rows:
        by_k[r["k"]].append(r["rmse"])
    ks = sorted(by_k)
    return ks, np.array([np.mean(by_k[k]) for k in ks]), np.array([np.std(by_k[k]) for k in ks])


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_dataset(dataset, runs):
    protocol = "kfold" if (dataset, "tsvd", "kfold") in runs else "random"
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=160)
    style(ax)
    all_ks = None
    for model, color in (("tsvd", BLUE), ("funk", ORANGE)):
        if (dataset, model, protocol) not in runs:
            continue
        ks, m, s = curve(runs[(dataset, model, protocol)])
        x = np.arange(len(ks))           # even spacing, K is not linear
        all_ks = ks
        ax.plot(x, m, color=color, linewidth=2, marker="o", markersize=4, label=NAMES[model])
        if s.any():
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)
        best = int(np.argmin(m))
        last = best == len(ks) - 1          # keep the label inside the plot
        ax.annotate(f"best K={ks[best]}: {m[best]:.4f}", (x[best], m[best]),
                    textcoords="offset points", xytext=(-6 if last else 0, -14),
                    ha="right" if last else "center", fontsize=8, color=INK)
    k5 = all_ks.index(5)
    ax.axvline(k5, color=MUTED, linestyle="--", linewidth=1)
    if dataset in PAPER:
        ax.plot([k5], [PAPER[dataset]], marker="D", markersize=7, color=INK, linestyle="none",
                label=f"paper SVD: {PAPER[dataset]}")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.08 * (hi - lo), hi)     # room for the bottom label
    ax.text(k5 + 0.1, ax.get_ylim()[0], " paper K=5", va="bottom", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(all_ks)))
    ax.set_xticklabels([str(k) for k in all_ks])
    ax.set_xlabel("K (number of factors)   K=0: no factors, baseline only", color=MUTED, fontsize=9)
    ax.set_ylabel("RMSE (lower is better)", color=MUTED, fontsize=9)
    split = "5-fold CV, mean and 1 std band" if protocol == "kfold" else "random 80/20, seed 42"
    ax.set_title(f"{TITLES[dataset]}\n{split}", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    path = os.path.join(OUT, "figures", f"k_rmse_{dataset}.png")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def plot_singular(sing):
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=160)
    style(ax)
    for dataset, color in (("mlsmall", BLUE), ("ml100k", ORANGE)):
        if dataset in sing:
            s = sing[dataset]
            ax.plot(np.arange(1, len(s) + 1), s, color=color, linewidth=2, label=TITLES[dataset])
    ax.set_xlabel("index k", color=MUTED, fontsize=9)
    ax.set_ylabel("singular value", color=MUTED, fontsize=9)
    ax.set_title("Singular values of the residual matrix (after the baseline)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "figures", "singular_values.png")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def table(dataset, runs):
    protocol = "kfold" if (dataset, "tsvd", "kfold") in runs else "random"
    print(f"\n### {TITLES[dataset]} ({protocol})\n")
    print("| K | Truncated SVD | Funk SVD |")
    print("|---|---|---|")
    cols = {}
    for model in ("tsvd", "funk"):
        if (dataset, model, protocol) in runs:
            ks, m, s = curve(runs[(dataset, model, protocol)])
            cols[model] = {k: (a, b) for k, a, b in zip(ks, m, s)}
    for k in sorted(set().union(*[c.keys() for c in cols.values()])):
        cells = []
        for model in ("tsvd", "funk"):
            if model in cols and k in cols[model]:
                a, b = cols[model][k]
                cells.append(f"{a:.4f}" + (f" ± {b:.4f}" if b else ""))
            else:
                cells.append("")
        print(f"| {k} | {cells[0]} | {cells[1]} |")


def main():
    runs, sing = load()
    for dataset in ("mlsmall", "ml100k", "ml20m"):
        if any(key[0] == dataset for key in runs):
            plot_dataset(dataset, runs)
            table(dataset, runs)
    plot_singular(sing)


if __name__ == "__main__":
    main()
