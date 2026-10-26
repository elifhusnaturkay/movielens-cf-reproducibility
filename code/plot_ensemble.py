"""Figures and tables for SVR, ensembles, deep learning and transformers.

    python3 plot_ensemble.py

Reads ../results/ensemble.jsonl, deep.jsonl and transformer.jsonl. Writes
results/figures/ensemble_chain.png, ensemble_models.png,
ensemble_temporal.png and ensemble_ml20m.png and prints the markdown tables.

For the transformers, the 3 seeds of one fold are averaged first, so every
model has one number per fold, like the other models.
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
STEPS = ["Traditional", "SVM", "Ensemble", "Deep learning", "Transformer"]
# same palette as the other figures, fixed order
STEP_COLOR = dict(zip(STEPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MODELS = [  # (key, name, step)
    ("funk", "Funk SVD", "Traditional"),
    ("cc", "Co-clustering (best basis)", "Traditional"),
    ("knn", "Item KNN", "Traditional"),
    ("svr", "SVR", "SVM"),
    ("stack_lr", "Stacking, linear", "Ensemble"),
    ("stack_xgb", "Stacking, XGBoost", "Ensemble"),
    ("bag_rf", "Bagging, Random Forest", "Ensemble"),
    ("boost_ada", "Boosting, AdaBoost", "Ensemble"),
    ("boost_xgb", "Boosting, XGBoost", "Ensemble"),
    ("boost_cat", "Boosting, CatBoost", "Ensemble"),
    ("ncf", "NCF", "Deep learning"),
    ("mlp", "MLP", "Deep learning"),
    ("hybrid", "NCF + MLP (hybrid)", "Deep learning"),
    ("bst", "Transformer, own (BST style)", "Transformer"),
    ("bert", "Transformer, HF BertModel", "Transformer"),
]
SHORT = {"stack_xgb": "Stacking (XGBoost)", "hybrid": "NCF + MLP", "bert": "HF BertModel",
         "bst": "own transformer"}
TITLES = {"mlsmall": "MovieLens latest small", "ml100k": "MovieLens 100k"}


def load():
    rows = []
    for name in ("ensemble", "deep"):
        rows += [json.loads(l) for l in open(os.path.join(RESULTS, name + ".jsonl"))]
    # transformer: mean over the seeds, one row per (dataset, split, fold, model)
    by = collections.defaultdict(list)
    for l in open(os.path.join(RESULTS, "transformer.jsonl")):
        r = json.loads(l)
        by[(r["dataset"], r["protocol"], r["fold"], r["model"])].append(r)
    for (ds, p, f, m), rs in by.items():
        rows.append(dict(dataset=ds, protocol=p, fold=f, model=m,
                         rmse=float(np.mean([r["rmse"] for r in rs])),
                         mae=float(np.mean([r["mae"] for r in rs]))))
    return rows


def score(rows, ds, model, protocol, val="rmse"):
    v = [r[val] for r in rows if r["dataset"] == ds and r["model"] == model
         and r["protocol"] == protocol]
    if not v:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v))


def style(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def best_per_step(rows, ds, protocol="kfold"):
    out = []
    for step in STEPS:
        keys = [k for k, _, s in MODELS if s == step
                and not np.isnan(score(rows, ds, k, protocol)[0])]
        k = min(keys, key=lambda k: score(rows, ds, k, protocol)[0])
        out.append((step, k, *score(rows, ds, k, protocol)))
    return out


def plot_chain(rows):
    panels = [("mlsmall", "kfold", "MovieLens latest small, 5-fold CV"),
              ("ml100k", "kfold", "MovieLens 100k, 5-fold CV"),
              ("ml20m", "random", "MovieLens 20M, random 80/20")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), dpi=160)
    names = {k: n for k, n, _ in MODELS}
    for ax, (ds, protocol, title) in zip(axes, panels):
        best = best_per_step(rows, ds, protocol)
        x = np.arange(len(best))
        y = [b[2] for b in best]
        ax.plot(x, y, color=GRID, linewidth=2, zorder=1)
        for xi, (step, k, m, s) in zip(x, best):
            ax.errorbar(xi, m, yerr=s if protocol == "kfold" else None, fmt="o",
                        color=STEP_COLOR[step], markersize=7, capsize=3, zorder=2)
            ax.annotate(f"{m:.3f}\n{SHORT.get(k, names[k])}", (xi, m),
                        xytext=(7, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=6.5, color=INK)
        ax.axhline(best[0][2], color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_xticks(x, [s.replace(" ", "\n") for s in STEPS])
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        lo, hi = min(y), max(y)
        ax.set_ylim(lo - 0.012, hi + 0.012)
        ax.set_xlim(-0.4, len(best) - 0.1)
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


def plot_temporal(rows, datasets, name, title_of):
    fig, axes = plt.subplots(1, len(datasets), dpi=160, sharey=True, squeeze=False,
                             figsize=(4.1 * len(datasets) + 1.4, 1.2 + 0.23 * len(MODELS)))
    for ax, ds in zip(axes[0], datasets):
        y = np.arange(len(MODELS))[::-1]
        for yi, (k, _, step) in zip(y, MODELS):
            a, b = score(rows, ds, k, "random")[0], score(rows, ds, k, "temporal")[0]
            if not (np.isnan(a) or np.isnan(b)):
                ax.plot([a, b], [yi, yi], color=GRID, linewidth=2, zorder=1)
            if not np.isnan(a):
                ax.plot(a, yi, "o", color=STEP_COLOR[step], markersize=5, zorder=2)
            if not np.isnan(b):
                ax.plot(b, yi, "o", markerfacecolor="white", markeredgecolor=STEP_COLOR[step],
                        markersize=5, zorder=2)
        ax.set_yticks(y, [n for _, n, _ in MODELS])
        ax.set_title(title_of(ds), fontsize=9, color=INK, loc="left")
        ax.set_xlabel("RMSE (filled = random 80/20, hollow = temporal 80/20)",
                      color=MUTED, fontsize=7.5)
        style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name))


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
    print("\nMovieLens 20M\n| Step | Model | random 80/20 | temporal 80/20 |")
    print("|---|---|---|---|")
    for k, name, step in MODELS:
        a, b = score(rows, "ml20m", k, "random")[0], score(rows, "ml20m", k, "temporal")[0]
        cell = lambda v: "not run" if np.isnan(v) else f"{v:.4f}"  # noqa: E731
        print(f"| {step} | {name} | {cell(a)} | {cell(b)} |")


if __name__ == "__main__":
    rows = load()
    plot_chain(rows)
    plot_all(rows)
    plot_temporal(rows, list(TITLES), "ensemble_temporal.png", TITLES.get)
    plot_temporal(rows, ["ml20m"], "ensemble_ml20m.png", lambda ds: "MovieLens 20M")
    tables(rows)
