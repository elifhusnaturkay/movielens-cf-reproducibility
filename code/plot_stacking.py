"""A picture of our stacking, so the two levels are clear.

    python3 plot_stacking.py

Level 1 is four different models (Funk SVD, co-clustering, KNN, SVR) on the
same data. Level 2 is XGBoost (stack_xgb) or a linear regression (stack_lr) on
their out-of-fold predictions, see stack.py. The numbers in the figure come
from ../results/ensemble.jsonl (5-fold mean).

Output: ../results/figures/stacking.png
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
import tema  # noqa: E402

tema.apply()
OUT = os.path.join(HERE, "..", "results", "figures")


def kfold_mean(model, ds):
    v = [r["rmse"] for r in map(json.loads, open(os.path.join(HERE, "..", "results", "ensemble.jsonl")))
         if r["model"] == model and r["dataset"] == ds and r["protocol"] == "kfold"]
    return np.mean(v)


def box(ax, x, y, w, h, title, sub, color, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=tema.edge(color), lw=1.6 if bold else 1.0))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=11.5,
            color=tema.INK, weight="bold")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.5, color=tema.INK)


def arrow(ax, a, b, color=tema.MUTED, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, lw=1.1,
                                 color=color, linestyle=ls, shrinkA=2, shrinkB=2))


# RMSE of each piece, latest-small and ml-100k
names = ["funk", "cc", "knn", "svr", "stack_lr", "stack_xgb"]
r = {m: (kfold_mean(m, "mlsmall"), kfold_mean(m, "ml100k")) for m in names}


def rm(m):
    return f"RMSE {r[m][0]:.3f} / {r[m][1]:.3f}"


fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.3)
ax.axis("off")

# level bands
ax.add_patch(FancyBboxPatch((2.75, 0.55), 3.5, 6.35, boxstyle="round,pad=0.02,rounding_size=0.2",
                            fc=tema.BG, ec="none"))
ax.add_patch(FancyBboxPatch((7.95, 1.6), 3.1, 4.25, boxstyle="round,pad=0.02,rounding_size=0.2",
                            fc=tema.BG, ec="none"))
ax.text(4.5, 7.05, "LEVEL 1: four different models", ha="center", fontsize=11, color=tema.OLIVE, weight="bold")
ax.text(4.5, 6.7, "the same data, the same question, different \"experts\"", ha="center",
        fontsize=9, color=tema.MUTED, style="italic")
ax.text(9.5, 6.0, "LEVEL 2: the meta model", ha="center", fontsize=11, color=tema.ROSE, weight="bold")
ax.text(9.5, 5.65, "the \"moderator\": learns how to combine them", ha="center",
        fontsize=9, color=tema.MUTED, style="italic")

# data
box(ax, 0.2, 2.9, 2.1, 1.6, "Data D", "train ratings of one split\n(user, movie, rating)", tema.CREAM)

# level 1
lvl1 = [("Funk SVD", "matrix factorization\n" + rm("funk"), tema.PISTACHIO),
        ("Co-clustering", "best basis: C3 10x5 / C4 50x2\n" + rm("cc"), tema.LIME),
        ("Item-based KNN", "KNNBaseline, k = 40\n" + rm("knn"), tema.SAGE),
        ("SVR (RBF)", "on 26-27 rating features\n" + rm("svr"), tema.LAVENDER)]
ys = [5.3, 3.85, 2.4, 0.95]
for (t, s, c), y in zip(lvl1, ys):
    box(ax, 3.0, y, 3.0, 1.15, t, s, c)
    arrow(ax, (2.3, 3.7), (3.0, y + 0.57))
    arrow(ax, (6.0, y + 0.57), (7.0, 3.7), color=tema.edge(c))

# predictions
ax.add_patch(FancyBboxPatch((6.95, 2.95), 0.75, 1.5, boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc="white", ec=tema.MUTED, lw=0.9))
for i, (lab, c) in enumerate(zip(["p1", "p2", "p3", "p4"], [x[2] for x in lvl1])):
    ax.text(7.32, 4.22 - i * 0.33, lab, ha="center", va="center", fontsize=9.5,
            color=tema.INK, bbox=dict(boxstyle="round,pad=0.15", fc=c, ec="none"))
ax.text(7.32, 2.65, "4 predictions\nper rating\n(out-of-fold)", ha="center", va="top", fontsize=8, color=tema.MUTED)

# features go only to XGBoost
ax.add_patch(FancyBboxPatch((6.7, 4.7), 1.25, 0.62, boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc="white", ec=tema.MUTED, lw=0.9, ls="--"))
ax.text(7.32, 5.01, "+ the same\n26-27 features", ha="center", va="center", fontsize=8.2, color=tema.INK)

# level 2
box(ax, 8.2, 3.95, 2.6, 1.3, "XGBoost", "on 4 predictions + features\n" + rm("stack_xgb"), tema.OLIVE, bold=True)
box(ax, 8.2, 1.95, 2.6, 1.2, "Linear regression", "on the 4 predictions only\n" + rm("stack_lr"), tema.PINK)
for t in ax.texts[-4:-2]:
    t.set_color("white")
arrow(ax, (7.7, 3.85), (8.2, 4.55))
arrow(ax, (7.7, 3.5), (8.2, 2.55))
arrow(ax, (7.95, 5.0), (8.2, 4.85), ls="--")

# output
box(ax, 11.4, 3.95, 1.45, 1.3, "rating", "XGBoost-based\nstacking", tema.CREAM, bold=True)
box(ax, 11.4, 1.95, 1.45, 1.2, "rating", "linear\nstacking", tema.CREAM)
arrow(ax, (10.8, 4.6), (11.4, 4.6), color=tema.OLIVE)
arrow(ax, (10.8, 2.55), (11.4, 2.55), color=tema.ROSE)

ax.text(9.5, 1.15, "The method is named after level 2:\nXGBoost on top = \"XGBoost-based stacking\".",
        ha="center", va="center", fontsize=9, color=tema.INK)
ax.text(0.2, 0.25, "RMSE = 5-fold mean on latest-small / ml-100k.  Out-of-fold: each base model predicts a train rating "
        "from a model fitted on the other 4/5 of the train set (5 folds inside train),\nso level 2 never sees a "
        "prediction made on a rating the base model was trained on. The test set is not used for any choice.",
        fontsize=8, color=tema.MUTED, va="bottom")
fig.suptitle("How our stacking works: four different models at level 1, XGBoost at level 2",
             fontsize=13.5, color=tema.INK, y=0.97)
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "stacking.png"), dpi=170, bbox_inches="tight", facecolor="white")
print("wrote stacking.png")
for m in names:
    print(f"{m:10s} {r[m][0]:.4f} {r[m][1]:.4f}")
