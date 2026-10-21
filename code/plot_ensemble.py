"""Figures and tables for SVR and stacking.

    python3 plot_ensemble.py

Reads ../results/ensemble.jsonl. Writes results/figures/ensemble_chain.png and
results/figures/ensemble_models.png and prints the markdown tables.
"""

import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "results", "figures")
os.makedirs(FIG, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STEPS = ["Traditional", "SVM", "Ensemble"]
# same palette as the week 5 figures, fixed order
STEP_COLOR = dict(zip(STEPS, ["#2a78d6", "#eb6834", "#1baf7a"]))
MODELS = [  # (key, name, step)
    ("funk", "Funk SVD", "Traditional"),
    ("cc", "Co-clustering (best basis)", "Traditional"),
    ("knn", "Item KNN", "Traditional"),
    ("svr", "SVR", "SVM"),
    ("stack_lr", "Stacking, linear", "Ensemble"),
    ("stack_xgb", "Stacking, XGBoost", "Ensemble"),
]
SHORT = {"stack_xgb": "Stacking (XGBoost)"}
TITLES = {"mlsmall": "MovieLens latest small", "ml100k": "MovieLens 100k"}


def load():
    return [json.loads(l) for l in open(os.path.join(HERE, "..", "results", "ensemble.jsonl"))]


def score(rows, ds, model, protocol, val="rmse"):
    v = [r[val] for r in rows if r["dataset"] == ds and r["model"] == model
         and r["protocol"] == protocol]
    return float(np.mean(v)), float(np.std(v))


def style(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def best_per_step(rows, ds):
    out = []
    for step in STEPS:
        keys = [k for k, _, s in MODELS if s == step]
        k = min(keys, key=lambda k: score(rows, ds, k, "kfold")[0])
        out.append((step, k, *score(rows, ds, k, "kfold")))
    return out


def plot_chain(rows):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), dpi=160)
    names = {k: n for k, n, _ in MODELS}
    for ax, ds in zip(axes, TITLES):
        best = best_per_step(rows, ds)
        x = np.arange(len(best))
        y = [b[2] for b in best]
        ax.plot(x, y, color=GRID, linewidth=2, zorder=1)
        for xi, (step, k, m, s) in zip(x, best):
            ax.errorbar(xi, m, yerr=s, fmt="o", color=STEP_COLOR[step], markersize=7,
                        capsize=3, zorder=2)
            ax.annotate(f"{m:.3f}\n{SHORT.get(k, names[k])}", (xi, m),
                        xytext=(7, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=6.5, color=INK)
        ax.axhline(best[0][2], color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_xticks(x, [s.replace(" ", "\n") for s in STEPS])
        ax.set_title(TITLES[ds] + ", 5-fold CV", fontsize=9, color=INK, loc="left")
        lo, hi = min(y), max(y)
        ax.set_ylim(lo - 0.012, hi + 0.012)
        ax.set_xlim(-0.4, len(best) - 0.2)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=7.5)
    axes[0].set_ylabel("RMSE of the best model (lower is better)", color=MUTED, fontsize=8)
    fig.suptitle("Best model of each step (dashed line = best traditional model)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ensemble_chain.png"))


def plot_all(rows):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 1.2 + 0.23 * len(MODELS)), dpi=160, sharey=True)
    for ax, ds in zip(axes, TITLES):
        y = np.arange(len(MODELS))[::-1]
        for yi, (k, name, step) in zip(y, MODELS):
            m, s = score(rows, ds, k, "kfold")
            ax.barh(yi, m, xerr=s, color=STEP_COLOR[step], height=0.7,
                    error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
            ax.text(m + s + 0.002, yi, f"{m:.3f}", va="center", fontsize=6.5, color=INK)
        vals = [score(rows, ds, k, "kfold")[0] for k, _, _ in MODELS]
        ax.set_xlim(min(vals) - 0.03, max(vals) + 0.02)
        ax.set_yticks(y, [n for _, n, _ in MODELS])
        ax.set_title(TITLES[ds] + ", 5-fold CV", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("RMSE (lower is better)", color=MUTED, fontsize=8)
        style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=STEP_COLOR[s]) for s in STEPS]
    fig.legend(handles, STEPS, loc="lower center", ncol=5, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(FIG, "ensemble_models.png"))


def tables(rows):
    names = {k: n for k, n, _ in MODELS}
    for ds in TITLES:
        print(f"\n{TITLES[ds]}\n| Step | Model | random 80/20 | temporal 80/20 | 5-fold CV |")
        print("|---|---|---|---|---|")
        for k, name, step in MODELS:
            a, b = score(rows, ds, k, "random")[0], score(rows, ds, k, "temporal")[0]
            m, s = score(rows, ds, k, "kfold")
            print(f"| {step} | {name} | {a:.4f} | {b:.4f} | {m:.4f} ± {s:.4f} |")
        print("\nbest per step:", [(s, names[k], round(m, 4), round(sd, 4))
                                   for s, k, m, sd in best_per_step(rows, ds)])


if __name__ == "__main__":
    rows = load()
    plot_chain(rows)
    plot_all(rows)
    tables(rows)
