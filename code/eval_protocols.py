"""Run one model on one dataset with one split protocol, save RMSE, MAE and R2.

    python3 eval_protocols.py ml100k svd random
    python3 eval_protocols.py ml20m  dt  kfold 3

datasets:  ml100k, mlsmall (paper's Dataset 1), ml20m (paper's Dataset 2)
protocols: random, temporal, kfold N, loo_last, loo_random (see common_split.py)
models:    svd      surprise.SVD, library defaults, seed 42
           own_svd  own_models.OwnSVD (same algorithm, our code)
           dt, dt20 DecisionTreeRegressor (default / max_depth=20), inputs as in
                    the paper: label-encoded user and movie ids + genres
           knn      surprise.KNNBaseline, item based, pearson_baseline, k=40
           cc       surprise.CoClustering 3x3
           own_cc   own_models.SparseCoClustering 3x3 (George & Merugu 2005)

The paper gives no settings for SVD or the tree, so we use library defaults.
Every run is appended to results/eval.jsonl.
"""

import json
import os
import resource
import sys
import time

import numpy as np

import common_split as cs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
JSONL = os.path.join(OUT, "eval.jsonl")


# datasets
def genres_ml100k():
    """(n_movies, 19) 0/1 matrix, row = 0 based movie id, from u.item."""
    path = os.path.join(os.path.dirname(cs.PATH), "u.item")
    rows = [line.rstrip("\n").split("|") for line in open(path, encoding="latin-1")]
    G = np.zeros((len(rows), 19), np.float32)
    for r in rows:
        G[int(r[0]) - 1] = [float(x) for x in r[-19:]]
    return G


def genres_mlsmall():
    """(max movie id + 1, 20) 0/1 matrix, row = movieId, from movies.csv."""
    import pandas as pd
    movies = pd.read_csv(os.path.join(cs.MLSMALL_DIR, "movies.csv"))
    names = sorted({g for gs in movies["genres"] for g in gs.split("|")})
    G = np.zeros((movies["movieId"].max() + 1, len(names)), np.float32)
    for mid, gs in zip(movies["movieId"], movies["genres"]):
        for g in gs.split("|"):
            G[mid, names.index(g)] = 1
    return G


def genres_ml20m():
    """(max movie id + 1, 20) 0/1 matrix, row = movieId, from movie.csv."""
    import pandas as pd
    movies = pd.read_csv(os.path.join(cs.ML20M_DIR, "movie.csv"))
    names = sorted({g for gs in movies["genres"] for g in gs.split("|")})
    G = np.zeros((movies["movieId"].max() + 1, len(names)), np.float32)
    for mid, gs in zip(movies["movieId"], movies["genres"]):
        for g in gs.split("|"):
            G[mid, names.index(g)] = 1
    return G


DATASETS = {
    # id_offset: add this to an id to get it as the file writes it (ml100k is
    # stored 0 based by common_split, the file is 1 based)
    "ml100k": dict(load=cs.load_ml100k_timed, genres=genres_ml100k, scale=(1, 5), id_offset=1),
    "ml20m": dict(load=cs.load_ml20m, genres=genres_ml20m, scale=(0.5, 5), id_offset=0),
    "mlsmall": dict(load=cs.load_mlsmall, genres=genres_mlsmall, scale=(0.5, 5), id_offset=0),
}


def get_split(raw, protocol, fold):
    if protocol == "random":
        return cs.split(raw)
    if protocol == "temporal":
        return cs.temporal_split(raw)
    if protocol == "kfold":
        return list(cs.kfold(raw, 5))[fold]
    if protocol == "loo_last":
        return cs.leave_one_out(raw, last=True)
    if protocol == "loo_random":
        return cs.leave_one_out(raw, last=False)
    sys.exit(f"unknown protocol {protocol}")


# models: (train, test, dataset) -> predictions
def fit_svd(train, test, ds):
    from surprise import SVD
    trainset, testset = cs.to_surprise(train, test, rating_scale=ds["scale"])
    algo = SVD(random_state=42).fit(trainset)
    return np.array([algo.predict(u, i).est for u, i, _ in testset])


def label_encode_as_text(ids):
    """Same codes as LabelEncoder on the ids as strings ("10" < "2")."""
    uniq = np.unique(ids)
    text_rank = np.empty(len(uniq), np.float32)
    text_rank[sorted(range(len(uniq)), key=lambda k: str(uniq[k]))] = np.arange(len(uniq))
    return text_rank[np.searchsorted(uniq, ids)]


def fit_tree(train, test, ds, max_depth=None):
    from sklearn.tree import DecisionTreeRegressor
    genres = ds["genres"]()
    rows = np.vstack([train, test])      # encode together: same id, same code
    X = np.empty((len(rows), 2 + genres.shape[1]), np.float32)
    X[:, 0] = label_encode_as_text(rows[:, 0].astype(np.int64) + ds["id_offset"])
    X[:, 1] = label_encode_as_text(rows[:, 1].astype(np.int64) + ds["id_offset"])
    X[:, 2:] = genres[rows[:, 1].astype(np.int64)]
    Xtr, Xte = X[:len(train)], X[len(train):]
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42).fit(Xtr, train[:, 2])
    print(f"  tree depth={tree.get_depth()} leaves={tree.get_n_leaves()}", flush=True)
    return tree.predict(Xte)


def fit_own_svd(train, test, ds):
    from own_models import OwnSVD
    return OwnSVD(rating_scale=ds["scale"], random_state=42).fit(train).predict(test)


def fit_cc(train, test, ds):
    from surprise import CoClustering
    trainset, testset = cs.to_surprise(train, test, rating_scale=ds["scale"])
    algo = CoClustering(n_cltr_u=3, n_cltr_i=3, random_state=42).fit(trainset)
    return np.array([algo.predict(u, i).est for u, i, _ in testset])


def fit_own_cc(train, test, ds):
    from own_models import SparseCoClustering
    both = np.vstack([train, test])
    shape = (int(both[:, 0].max()) + 1, int(both[:, 1].max()) + 1)
    model = SparseCoClustering(3, 3, max_iter=50, random_state=42).fit(train, shape)
    print(f"  own_cc iterations={model.n_iter}", flush=True)
    return model.predict(test)


def fit_knn(train, test, ds):
    # item based, pearson_baseline similarity, k=40
    from surprise import KNNBaseline
    trainset, testset = cs.to_surprise(train, test, rating_scale=ds["scale"])
    algo = KNNBaseline(k=40, sim_options={"name": "pearson_baseline", "user_based": False},
                       verbose=False).fit(trainset)
    return np.array([algo.predict(u, i).est for u, i, _ in testset])


MODELS = {
    "svd": fit_svd,
    "own_svd": fit_own_svd,
    "cc": fit_cc,
    "own_cc": fit_own_cc,
    "dt": fit_tree,
    "dt20": lambda a, b, ds: fit_tree(a, b, ds, max_depth=20),
    "knn": fit_knn,
}


def main():
    dataset, model, protocol = sys.argv[1:4]
    fold = int(sys.argv[4]) if len(sys.argv) > 4 else None
    ds = DATASETS[dataset]

    raw = ds["load"]()
    train, test = get_split(raw, protocol, fold)
    del raw
    fp = cs.fingerprint(test)
    print(f"[split] eval_protocols.py  {dataset}  {model}  {protocol}"
          f"{'' if fold is None else f' fold={fold}'}  seed={cs.SEED}  train={len(train)}  "
          f"test={len(test)}  fingerprint={fp}", flush=True)

    unseen = int((~np.isin(test[:, 1], train[:, 1])).sum())
    t = time.time()
    pred = MODELS[model](train, test, ds)
    fit_s = time.time() - t

    y = test[:, 2]
    err = pred - y
    row = dict(
        dataset=dataset, model=model, protocol=protocol, fold=fold,
        n_train=len(train), n_test=len(test), fingerprint=fp,
        rmse=float(np.sqrt(np.mean(err ** 2))),
        mae=float(np.mean(np.abs(err))),
        r2=float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
        test_var=float(y.var()),
        unseen_items=unseen,
        fit_s=round(fit_s, 1),
        peak_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2),
    )
    print(json.dumps(row), flush=True)
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
