"""Predictions of the single models, saved for stacking.

    python3 base_preds.py mlsmall random
    python3 base_preds.py ml100k kfold 0

For one dataset and one split it saves, for every base model:
  test  model trained on the full train set, predicts the test set
  oof   out-of-fold predictions on the train set (5 folds inside train)

The meta model in stack.py learns only from oof, so it never sees a rating
that the base model trained on.

Base models:
  funk   Funk SVD, our own code (= surprise.SVD), library defaults
  cc     best co-clustering basis of the dataset (C3 10x5 / C4 50x2 / C4 50x50)
  knn    Surprise KNNBaseline, item based, pearson_baseline, k=40
         (on 20M our own copy in own_knn.py, Surprise does not fit in memory)
  svr    RBF SVR on features.py, C=0.3, epsilon=0.3 (chosen on a validation
         part of the latest-small train set, not on the test set)
         (on 20M a Nystroem approximation, see fit_svr_big)

It also saves the features (out-of-fold for train rows, see oof_features),
so stack.py, dl.py and the boosting models use exactly the same inputs.

Output: ../results/preds/preds_<dataset>_<split>.npz and one line per model
in ../results/ensemble.jsonl.
"""

import json
import os
import sys
import time

import numpy as np

import common_split as cs
import eval_protocols as ep
from bases import BasisCoClustering
from features import Features
from own_models import OwnSVD

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
PREDS = os.path.join(RESULTS, "preds")
os.makedirs(PREDS, exist_ok=True)

DATASETS = {
    "mlsmall": dict(load=cs.load_mlsmall, genres=ep.genres_mlsmall, scale=(0.5, 5)),
    "ml100k": dict(load=cs.load_ml100k_timed, genres=ep.genres_ml100k, scale=(1, 5)),
    "ml20m": dict(load=cs.load_ml20m, genres=ep.genres_ml20m, scale=(0.5, 5), big=True),
}
BEST_CC = {"mlsmall": ("C3", 10, 5), "ml100k": ("C4", 50, 2), "ml20m": ("C4", 50, 50)}


def get_split(raw, protocol, fold):
    if protocol == "random":
        return cs.split(raw)
    if protocol == "temporal":
        return cs.temporal_split(raw)
    if protocol == "kfold":
        return list(cs.kfold(raw, 5))[fold]
    sys.exit(f"unknown split {protocol}")


def oof_features(train, test, ds, shape):
    """Train rows get features from the other 4/5 of train (no leakage),
    test rows get features from the full train set."""
    Xtr = np.zeros((len(train), 7 + ds["G"].shape[1]), np.float32)
    inner = np.random.default_rng(7).permutation(len(train))
    for part in np.array_split(inner, 5):
        mask = np.zeros(len(train), bool)
        mask[part] = True
        Xtr[mask] = Features(ds["G"]).fit(train[~mask], shape).transform(train[mask])
    Xte = Features(ds["G"]).fit(train, shape).transform(test)
    return Xtr, Xte


def fit_funk(train, test, ds, shape):
    return OwnSVD(rating_scale=ds["scale"], random_state=42).fit(train).predict(test)


def fit_cc(train, test, ds, shape):
    basis, k, l = ds["cc"]
    model = BasisCoClustering(basis, k, l, max_iter=50, random_state=42).fit(train, shape)
    return np.clip(model.predict(test), *ds["scale"])


def fit_knn(train, test, ds, shape):
    if ds.get("big"):
        # same predictions as Surprise (max difference 1.3e-15 on latest-small and 100k)
        from own_knn import OwnKNN
        return OwnKNN(rating_scale=ds["scale"], dtype=np.float32).fit(train).predict(test)
    from surprise import KNNBaseline
    trainset, testset = cs.to_surprise(train, test, rating_scale=ds["scale"])
    algo = KNNBaseline(k=40, sim_options={"name": "pearson_baseline", "user_based": False},
                       verbose=False).fit(trainset)
    return np.array([algo.predict(u, i).est for u, i, _ in testset])


def fit_svr(train, test, ds, shape):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    Xtr, Xte = oof_features(train, test, ds, shape)
    if ds.get("big"):
        return fit_svr_big(train, Xtr, Xte, ds)
    sc = StandardScaler().fit(Xtr)
    model = SVR(C=0.3, epsilon=0.3, cache_size=2000).fit(sc.transform(Xtr), train[:, 2])
    return np.clip(model.predict(sc.transform(Xte)), *ds["scale"])


def fit_svr_big(train, Xtr, Xte, ds):
    """20M: an exact RBF SVR is not possible with 16 million ratings. The RBF
    kernel is approximated with Nystroem (1000 components) and a linear SVR is
    trained on a random sample of 500,000 train ratings. Same C, epsilon and
    gamma as the exact SVR. On latest-small (all train ratings) this gives
    0.8475, the exact SVR 0.8442."""
    from sklearn.kernel_approximation import Nystroem
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVR
    sc = StandardScaler().fit(Xtr)
    pick = np.random.default_rng(42).choice(len(Xtr), min(500_000, len(Xtr)), replace=False)
    A = sc.transform(Xtr[pick])
    ny = Nystroem(gamma=1 / (A.shape[1] * A.var()), n_components=1000, random_state=42).fit(A)
    model = LinearSVR(C=0.3, epsilon=0.3, random_state=42, max_iter=5000)
    model.fit(ny.transform(A), train[pick, 2])
    out = np.empty(len(Xte))
    for s in range(0, len(Xte), 200_000):                # in parts, to save memory
        out[s:s + 200_000] = model.predict(ny.transform(sc.transform(Xte[s:s + 200_000])))
    return np.clip(out, *ds["scale"])


MODELS = {"funk": fit_funk, "cc": fit_cc, "knn": fit_knn, "svr": fit_svr}


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def main():
    dataset, protocol = sys.argv[1], sys.argv[2]
    fold = int(sys.argv[3]) if len(sys.argv) > 3 else None
    ds = dict(DATASETS[dataset], cc=BEST_CC[dataset])
    ds["G"] = ds["genres"]()
    raw = ds["load"]()
    shape = (int(raw[:, 0].max()) + 1, int(raw[:, 1].max()) + 1)
    train, test = get_split(raw, protocol, fold)
    fp = cs.fingerprint(test)
    print(f"{dataset} {protocol} {fold}  train={len(train)} test={len(test)} fingerprint={fp}",
          flush=True)

    inner = np.random.default_rng(7).permutation(len(train))
    parts = np.array_split(inner, 5)

    Xtr, Xte = oof_features(train, test, ds, shape)
    saved = {"y_train": train[:, 2], "y_test": test[:, 2], "fingerprint": fp,
             "X_train": Xtr, "X_test": Xte, "train": train[:, :3], "test": test[:, :3],
             "shape": np.array(shape), "scale": np.array(ds["scale"])}
    for name, fit in MODELS.items():
        t = time.time()
        oof = np.zeros(len(train))
        for part in parts:
            mask = np.zeros(len(train), bool)
            mask[part] = True
            oof[mask] = fit(train[~mask], train[mask], ds, shape)
        pred = fit(train, test, ds, shape)
        saved["oof_" + name] = oof
        saved["test_" + name] = pred
        sec = time.time() - t
        print(f"  {name:5s} test RMSE {rmse(pred, test[:, 2]):.4f}   ({sec:.0f} s)", flush=True)
        with open(os.path.join(RESULTS, "ensemble.jsonl"), "a") as f:
            f.write(json.dumps(dict(dataset=dataset, protocol=protocol, fold=fold, model=name,
                                    fingerprint=fp, rmse=rmse(pred, test[:, 2]),
                                    mae=mae(pred, test[:, 2]), seconds=round(sec, 1))) + "\n")

    tag = protocol if fold is None else f"{protocol}{fold}"
    np.savez(os.path.join(PREDS, f"preds_{dataset}_{tag}.npz"), **saved)


if __name__ == "__main__":
    main()
