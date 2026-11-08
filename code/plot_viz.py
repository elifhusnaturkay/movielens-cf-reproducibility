"""The four summary figures of the visualization step.

    python3 plot_viz.py

Reads ../results/viz/*_maps.npz and ../results/viz.jsonl written by viz.py, so
viz.py has to run first. Writes ../results/figures/viz_*.png.
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "figures")
os.makedirs(OUT, exist_ok=True)
import tema  # noqa: E402
from tema import OLIVE, PINK, LAVENDER, LIME, CREAM, ROSE, SAGE  # noqa: E402

tema.apply()
plt.rcParams.update({"font.size": 10})
GRAY = tema.OTHER
EN = {"female": "female", "male": "male", "other": "other"}


def load(ds, side):
    z = np.load(os.path.join(HERE, "..", "results", "viz", f"{ds}_{side}_maps.npz"))
    return {k: z[k] for k in z.files}


def clean(ax, title):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9.5)


def num(fig, ax, Y, v, label):
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=v, s=5, cmap=tema.GREENS_UP, alpha=0.85, lw=0)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label(label, fontsize=8)


def cat(ax, Y, v, colors, rest=GRAY):
    """colors: {category: color}; every other category is drawn gray first"""
    other = ~np.isin(v, list(colors))
    if other.any():
        ax.scatter(Y[other, 0], Y[other, 1], s=5, color=rest, alpha=0.6, lw=0, label=f"other ({other.sum()})")
    for c, col in colors.items():
        m = v == c
        ax.scatter(Y[m, 0], Y[m, 1], s=6, color=col, alpha=0.85, lw=0, label=f"{EN.get(c, c)} ({m.sum()})")
    ax.legend(fontsize=7, markerscale=2.5, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# 1) latest small users: four methods, raw and centered, colored by number of ratings
def methods():
    d = load("mlsmall", "users")
    v = np.log10(d["info|count"])
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
    for r, (fill, fname) in enumerate([("raw", "raw"), ("centered", "user mean removed")]):
        for c, m in enumerate(["SVD", "PCA", "t-SNE p30", "UMAP"]):
            ax = axes[r, c]
            sc = ax.scatter(d[f"{fill}|{m}"][:, 0], d[f"{fill}|{m}"][:, 1], c=v, s=6, cmap=tema.GREENS_UP, alpha=0.85, lw=0)
            clean(ax, f"{m}, {fname}")
    fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01).set_label("number of ratings (log10)")
    fig.suptitle("Latest small, 566 users. Color = how many movies the user rated", fontsize=11.5)
    save(fig, "viz_methods.png")


# 2) ml-100k users, user mean removed: does anything we know explain the map?
def users_100k():
    d = load("ml100k", "users")
    Y = d["centered|UMAP"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    fig.subplots_adjust(wspace=0.6)
    num(fig, axes[0], Y, np.log10(d["info|count"]), "number of ratings (log10)")
    clean(axes[0], "number of ratings")
    cat(axes[1], Y, d["cat|age"], {"<25": LIME, "25-34": OLIVE, "35-49": PINK, "50+": LAVENDER})
    clean(axes[1], "age")
    cat(axes[2], Y, d["cat|gender"], {"male": OLIVE, "female": PINK})
    clean(axes[2], "gender")
    cat(axes[3], Y, d["cat|occupation"], {"student": OLIVE, "educator": PINK, "programmer": LAVENDER,
                                      "engineer": CREAM, "administrator": LIME})
    clean(axes[3], "occupation (5 largest)")
    fig.suptitle("100k, 917 users, UMAP after removing each user's mean", fontsize=11.5, y=1.04)
    save(fig, "viz_users_100k.png")


# 3) 20M movies: genre islands (raw) and the average rating axis (user mean removed)
def movies_20m():
    d = load("ml20m", "movies")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    fig.subplots_adjust(wspace=0.5)
    cat(axes[0], d["raw|UMAP"], d["cat|main genre"], {"Animation": PINK, "Horror": OLIVE, "Sci-Fi": LAVENDER})
    clean(axes[0], "UMAP, raw. Color = main genre")
    num(fig, axes[1], d["centered|UMAP"], d["info|mean"], "average rating of the movie")
    clean(axes[1], "UMAP, user mean removed. Color = average rating")
    fig.suptitle("20M sample, 4,442 movies", fontsize=11.5, y=1.02)
    save(fig, "viz_movies_20m.png")


# 4) silhouette against K, user mean removed
def silhouette():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "..", "results", "viz.jsonl"))]
    ks = list(range(2, 16))
    names = {"mlsmall": "Latest small", "ml100k": "100k", "ml20m": "20M sample"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.8), sharey=True)
    fig.subplots_adjust(hspace=0.35)
    for c, ds in enumerate(names):
        for r, side in enumerate(["users", "movies"]):
            ax = axes[r, c]
            for m, col in [("PCA-50", SAGE), ("t-SNE p30", OLIVE), ("UMAP", PINK), ("SVD", LAVENDER)]:
                x = [l for l in rows if l.get("ds") == ds and l.get("side") == side and l.get("fill") == "centered"
                     and l.get("method") == m][0]
                ax.plot(ks, x["sil"], "o-", ms=3.5, lw=2, color=col, label=m)
            ax.set_title(f"{names[ds]}, {side}", fontsize=10); tema.style(ax, "both"); ax.set_xticks(ks[::2])
            if r == 1:
                ax.set_xlabel("K (number of K-means clusters)")
            if c == 0:
                ax.set_ylabel("silhouette")
    axes[0, 2].legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle("Silhouette score against K, user mean removed. Higher = clearer groups", fontsize=11.5)
    save(fig, "viz_silhouette.png")


if __name__ == "__main__":
    methods()
    users_100k()
    movies_20m()
    silhouette()
