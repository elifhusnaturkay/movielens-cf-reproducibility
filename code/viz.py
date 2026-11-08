"""Look at the data before the numbers: how many user and movie groups are there?

    python3 viz.py mlsmall
    python3 viz.py ml100k
    python3 viz.py ml20m        (random 5,000 users)

Every user (and every movie) is a point in a very high dimensional space
(one dimension per movie, or per user). We bring it down to 2D with four
methods and look how many groups there are:

    SVD     U[:, :2] * sigma, straight on the sparse matrix
    PCA     the first 2 principal components
    t-SNE   on the first 50 principal components, perplexity 5, 30 and 50
    UMAP    on the first 50 principal components, default settings

Choices:
    - all ratings are used, not only the train part (train_check.py repeats
      the silhouette scores with train ratings only)
    - users and movies with fewer than 20 ratings are left out
    - the empty cells are 0 in two ways, both are drawn:
        "raw"       raw stars, empty = 0
        "centered"  each user's own mean is subtracted first, empty = 0 (neutral)
    - number of groups: by eye and by the silhouette score of K-means, K = 2..15

Output: ../results/viz/<ds>_*.png and <ds>_<side>_maps.npz, and one line per
(side, fill, method) in ../results/viz.jsonl.
"""

import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap

HERE = os.path.dirname(os.path.abspath(__file__))
import common_split as cs  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "viz")
os.makedirs(OUT, exist_ok=True)

SEED = 42
MIN_RATINGS = 20
N_USERS_20M = 5000
PERPLEXITIES = [5, 30, 50]
KS = list(range(2, 16))
FILLS = ["raw", "centered"]   # raw stars / user mean subtracted

# A movie has many genres and the files list them in alphabetical order, so the
# first one is almost always Action or Adventure. We take the first genre of this
# list that the movie has: the more specific genres come first.
GENRE_ORDER = ["Animation", "Documentary", "Horror", "Sci-Fi", "Musical", "War", "Western",
               "Film-Noir", "Children", "Crime", "Mystery", "Fantasy", "Romance", "Thriller",
               "Action", "Adventure", "Comedy", "Drama"]
ML100K_GENRES = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                 "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                 "Romance", "Sci-Fi", "Thriller", "War", "Western"]
COLORS = ["#2b5fa8", "#e07b39", "#2a9d6f", "#c0392b", "#7c3aed", "#d4a017", "#17a2b8",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#1f3b63", "#ff9896", "#98df8a", "#c5b0d5"]


# ---------------------------------------------------------------- data

def main_genre(genres):
    for g in GENRE_ORDER:
        if g in genres:
            return g
    return "other"


def load_genres(ds):
    """movie id (as in the ratings file) -> main genre"""
    out = {}
    if ds == "ml100k":
        path = os.path.join(os.path.dirname(cs.PATH), "u.item")
        with open(path, encoding="latin-1") as f:
            for line in f:
                p = line.rstrip("\n").split("|")
                flags = [int(x) for x in p[5:24]]
                gs = [ML100K_GENRES[i] for i, x in enumerate(flags) if x == 1]
                out[int(p[0]) - 1] = main_genre(gs)      # load() makes ids 0 based
        return out
    import pandas as pd
    if ds == "mlsmall":
        df = pd.read_csv(os.path.join(cs.MLSMALL_DIR, "movies.csv"))
    else:
        df = pd.read_csv(os.path.join(cs.ML20M_DIR, "movie.csv"))
    for mid, g in zip(df["movieId"], df["genres"]):
        out[int(mid)] = main_genre(g.replace("Children's", "Children").split("|"))
    return out


def load_ml100k_users():
    """user id (0 based) -> (age, gender, occupation)"""
    path = os.path.join(os.path.dirname(cs.PATH), "u.user")
    out = {}
    with open(path) as f:
        for line in f:
            p = line.strip().split("|")
            out[int(p[0]) - 1] = (int(p[1]), p[2], p[3])
    return out


def load_ratings(ds):
    if ds == "mlsmall":
        return cs.load_mlsmall()[:, :3]
    if ds == "ml100k":
        return cs.load().astype(float)
    raw = cs.load_ml20m()[:, :3]
    rng = np.random.default_rng(SEED)
    users = np.unique(raw[:, 0])
    keep = rng.choice(users, N_USERS_20M, replace=False)
    return raw[np.isin(raw[:, 0], keep)]


def filter_min(r):
    """drop users and movies with fewer than MIN_RATINGS ratings, until nothing changes"""
    while True:
        u, cu = np.unique(r[:, 0], return_counts=True)
        m, cm = np.unique(r[:, 1], return_counts=True)
        bad_u = u[cu < MIN_RATINGS]
        bad_m = m[cm < MIN_RATINGS]
        if len(bad_u) == 0 and len(bad_m) == 0:
            return r
        r = r[~np.isin(r[:, 0], bad_u) & ~np.isin(r[:, 1], bad_m)]


def build_matrix(r):
    """sparse users x movies matrix, plus the original ids of rows and columns"""
    uid, ui = np.unique(r[:, 0], return_inverse=True)
    mid, mi = np.unique(r[:, 1], return_inverse=True)
    A = sp.csr_matrix((r[:, 2], (ui, mi)), shape=(len(uid), len(mid)))
    return A, uid.astype(int), mid.astype(int)


def center_users(A):
    """subtract each user's own mean from his known ratings, empty cells stay 0"""
    A = A.tocsr(copy=True)
    counts = np.diff(A.indptr)
    means = np.asarray(A.sum(1)).ravel() / counts
    A.data -= np.repeat(means, counts)
    return A


# ---------------------------------------------------------------- 2D maps

def svd_map(A):
    """The SVD map: the first 2 columns of U times sigma (rows of A)."""
    U, s, Vt = svds(A.astype(float), k=2)
    order = np.argsort(s)[::-1]                 # svds gives the smallest first
    return U[:, order] * s[order]


def maps_for(X):
    """X: sparse, one row per point. Returns {method: (n, 2) array} and the seconds."""
    maps, secs = {}, {}
    t = time.time(); maps["SVD"] = svd_map(X); secs["SVD"] = time.time() - t
    dense = X.toarray()
    t = time.time()
    p50 = PCA(n_components=min(50, min(dense.shape) - 1), random_state=SEED).fit_transform(dense)
    secs["PCA"] = time.time() - t
    maps["PCA"] = p50[:, :2]
    maps["PCA-50"] = p50                        # not drawn, only for the silhouette
    for p in PERPLEXITIES:
        t = time.time()
        maps[f"t-SNE p{p}"] = TSNE(n_components=2, perplexity=p, init="pca",
                                   random_state=SEED).fit_transform(p50)
        secs[f"t-SNE p{p}"] = time.time() - t
    t = time.time()
    maps["UMAP"] = umap.UMAP(random_state=SEED).fit_transform(p50)
    secs["UMAP"] = time.time() - t
    return maps, secs


def silhouettes(Y):
    """silhouette of K-means for K = 2..15 on the points Y"""
    out = []
    for k in KS:
        lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Y)
        out.append(silhouette_score(Y, lab, sample_size=min(len(Y), 3000), random_state=SEED))
    return out


def best_labels(Y, sil):
    k = KS[int(np.argmax(sil))]
    return k, KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Y)


# ---------------------------------------------------------------- drawing

def scatter_cat(ax, Y, lab, names=None, legend=False, s=6):
    cats = sorted(set(lab), key=lambda c: (list(lab).count(c) * -1))
    for i, c in enumerate(cats):
        m = np.array([x == c for x in lab])
        col = "#c8cdd3" if c == "other" else COLORS[i % len(COLORS)]
        ax.scatter(Y[m, 0], Y[m, 1], s=s, color=col, alpha=0.75, lw=0,
                   label=f"{c} ({m.sum()})" if names is None else names[c])
    if legend:
        ax.legend(fontsize=7, markerscale=2.5, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))


def scatter_num(fig, ax, Y, v, label, s=6):
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=v, s=s, cmap="viridis", alpha=0.8, lw=0)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label(label, fontsize=8)


def clean(ax, title):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", name, flush=True)


# ---------------------------------------------------------------- main

DS_NAME = {"mlsmall": "latest-small", "ml100k": "ml-100k", "ml20m": "20M (5,000 users)"}
SIDE_NAME = {"users": "users", "movies": "movies"}


def run_side(ds, side, A, C, info, log):
    """A (raw) and C (user centered): sparse, one row per point of this side"""
    name = f"{DS_NAME[ds]}, {SIDE_NAME[side]} ({A.shape[0]} nokta)"
    print(f"{ds} {side}: {A.shape}", flush=True)
    res = {}
    for fill, X in [("raw", A), ("centered", C)]:
        maps, secs = maps_for(X)
        sil = {m: silhouettes(Y) for m, Y in maps.items()}
        res[fill] = (maps, sil)
        for m in maps:
            k = KS[int(np.argmax(sil[m]))]
            line = {"ds": ds, "side": side, "fill": fill, "method": m, "n": int(A.shape[0]),
                    "best_k": k, "best_sil": round(float(max(sil[m])), 4),
                    "sil": [round(float(x), 4) for x in sil[m]], "secs": round(secs.get(m, 0.0), 1)}
            log.write(json.dumps(line) + "\n"); log.flush()
            print(f"  {fill:6s} {m:10s} best K={k:2d} silhouette={max(sil[m]):.3f}", flush=True)

    # save the 2D maps and the point info, so the report figures can be drawn without running again
    keep = {}
    for fill in FILLS:
        for m, Y in res[fill][0].items():
            if m != "PCA-50":
                keep[f"{fill}|{m}"] = Y
    keep["info|count"] = info["count"]; keep["info|mean"] = info["mean"]
    for k, v in info.get("cats", []):
        keep[f"cat|{k}"] = v
    np.savez_compressed(os.path.join(OUT, f"{ds}_{side}_maps.npz"), **keep)

    # 1) four methods x two fills, colored by the best K-means
    methods = ["SVD", "PCA", "t-SNE p30", "UMAP"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for r, fill in enumerate(FILLS):
        maps, sil = res[fill]
        for c, m in enumerate(methods):
            k, lab = best_labels(maps[m], sil[m])
            scatter_cat(axes[r, c], maps[m], lab)
            clean(axes[r, c], f"{fill} · {m}\nen iyi K = {k}, silhouette = {max(sil[m]):.2f}")
    fig.suptitle(f"{name}: four methods, color = best K-means clustering by silhouette", fontsize=12)
    save(fig, f"{ds}_{side}_1_methods.png")

    # 2) t-SNE with three perplexities
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for r, fill in enumerate(FILLS):
        maps, sil = res[fill]
        for c, p in enumerate(PERPLEXITIES):
            m = f"t-SNE p{p}"
            k, lab = best_labels(maps[m], sil[m])
            scatter_cat(axes[r, c], maps[m], lab)
            clean(axes[r, c], f"{fill} · perplexity {p}\nen iyi K = {k}, silhouette = {max(sil[m]):.2f}")
    fig.suptitle(f"{name}: t-SNE, three perplexities. Groups seen in all three do not come from the setting", fontsize=12)
    save(fig, f"{ds}_{side}_2_perplexity.png")

    # 3) silhouette curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, fill in zip(axes, FILLS):
        maps, sil = res[fill]
        for i, m in enumerate(["SVD", "PCA", "PCA-50", "t-SNE p5", "t-SNE p30", "t-SNE p50", "UMAP"]):
            ax.plot(KS, sil[m], "o-", ms=3, color=COLORS[i], label=m)
        ax.set_xticks(KS); ax.set_xlabel("K (number of K-means clusters)"); ax.set_title(fill, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("silhouette (higher = clearer groups)")
    axes[1].legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle(f"{name}: silhouette against K", fontsize=12)
    save(fig, f"{ds}_{side}_3_silhouette.png")

    # 4) the same maps colored by what we know about the points
    panels = [("number of ratings (log10)", np.log10(info["count"])), ("mean rating", info["mean"])]
    panels += [(k, v) for k, v in info.get("cats", [])]
    rows = [(f, m) for f in FILLS for m in ["t-SNE p30", "UMAP"]]
    fig, axes = plt.subplots(len(rows), len(panels), figsize=(5.2 * len(panels), 4 * len(rows)))
    fig.subplots_adjust(wspace=0.55, hspace=0.25)
    for c, (lab, v) in enumerate(panels):
        for r, (fill, m) in enumerate(rows):
            ax = axes[r, c]
            Y = res[fill][0][m]
            if v.dtype.kind == "f":
                scatter_num(fig, ax, Y, v, lab)
            else:
                scatter_cat(ax, Y, v, legend=True)
            clean(ax, f"{fill} · {m} · color: {lab}")
    fig.suptitle(f"{name}: what are the groups about? (top two rows raw, bottom two rows centered)", fontsize=12, y=0.92)
    save(fig, f"{ds}_{side}_4_colors.png")


def main():
    ds = sys.argv[1]
    t0 = time.time()
    raw = load_ratings(ds)
    r = filter_min(raw)
    A, uid, mid = build_matrix(r)
    print(f"{ds}: {len(raw)} ratings, {len(np.unique(raw[:, 0]))} users, {len(np.unique(raw[:, 1]))} movies -> "
          f"after the filter {len(r)} ratings, {A.shape[0]} users, {A.shape[1]} movies", flush=True)
    C = center_users(A)
    genre = load_genres(ds)
    filt = {"ds": ds, "filter": True, "ratings_before": int(len(raw)),
            "users_before": int(len(np.unique(raw[:, 0]))), "movies_before": int(len(np.unique(raw[:, 1]))),
            "ratings_after": int(len(r)), "users_after": int(A.shape[0]), "movies_after": int(A.shape[1]),
            "density_after": round(len(r) / (A.shape[0] * A.shape[1]), 4)}

    log = open(os.path.join(RESULTS, "viz.jsonl"), "a")
    log.write(json.dumps(filt) + "\n")

    def side_info(M):
        cnt = np.diff(M.tocsr().indptr).astype(float)
        return {"count": cnt, "mean": np.asarray(M.sum(1)).ravel() / cnt}

    # users
    info = side_info(A)
    if ds == "ml100k":
        dem = load_ml100k_users()
        age = np.array([dem[u][0] for u in uid])
        bins = np.array(["<25" if a < 25 else "25-34" if a < 35 else "35-49" if a < 50 else "50+" for a in age])
        gender = np.array(["female" if dem[u][1] == "F" else "male" for u in uid])
        occ = np.array([dem[u][2] for u in uid])
        top = [o for o, _ in sorted({o: (occ == o).sum() for o in set(occ)}.items(), key=lambda x: -x[1])[:7]]
        occ = np.array([o if o in top else "other" for o in occ])
        info["cats"] = [("age", bins), ("gender", gender), ("occupation", occ)]
    run_side(ds, "users", A, C, info, log)

    # movies: the same two matrices, transposed
    info = side_info(A.T)
    g = np.array([genre.get(m, "other") for m in mid])
    top = [x for x, _ in sorted({x: (g == x).sum() for x in set(g)}.items(), key=lambda x: -x[1])[:10]]
    g = np.array([x if x in top else "other" for x in g])
    info["cats"] = [("main genre", g)]
    run_side(ds, "movies", A.T.tocsr(), C.T.tocsr(), info, log)
    log.close()
    print(f"{ds} bitti: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
