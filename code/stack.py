"""Stacking, boosting and bagging.

    python3 stack.py mlsmall random
    python3 stack.py ml100k kfold 0

Reads ../results/preds/preds_<dataset>_<split>.npz (made by base_preds.py).

Stacking of Funk SVD, co-clustering, KNN and SVR (the meta model learns from
the out-of-fold predictions only):
  stack_lr     linear regression on [funk, cc, knn, svr]
  stack_xgb    XGBoost on [funk, cc, knn, svr] + the features
Boosting and bagging (on the features only, like the SVR):
  bag_rf       Random Forest (bagging of trees)
  boost_ada    AdaBoost (trees of depth 6)
  boost_xgb    XGBoost
  boost_cat    CatBoost

Settings are fixed by hand, common values, not tuned on the test set.
On 20M (more than 1M train ratings) the Random Forest trees see 800,000
ratings each and AdaBoost is trained on a random 2M sample, else they
take too long.
Results go to ../results/ensemble.jsonl.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
BASE = ["funk", "cc", "knn", "svr"]


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def models(big=False):
    from catboost import CatBoostRegressor
    from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    def xgb():
        return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                            colsample_bytree=0.8, random_state=42, n_jobs=8)

    # name -> (inputs, model)
    return {
        "stack_lr": ("preds", LinearRegression()),
        "stack_xgb": ("preds+features", xgb()),
        "bag_rf": ("features", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=10, max_features=0.5,
            max_samples=800_000 if big else None,     # 20M: each tree sees 800k ratings
            random_state=42, n_jobs=8)),
        "boost_ada": ("features", AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=6), n_estimators=100, learning_rate=0.1,
            random_state=42)),
        "boost_xgb": ("features", xgb()),
        "boost_cat": ("features", CatBoostRegressor(
            iterations=800, learning_rate=0.05, depth=6, random_seed=42,
            verbose=0, thread_count=8, allow_writing_files=False)),
    }


def main():
    dataset, protocol = sys.argv[1], sys.argv[2]
    fold = int(sys.argv[3]) if len(sys.argv) > 3 else None
    tag = protocol if fold is None else f"{protocol}{fold}"
    d = np.load(os.path.join(RESULTS, "preds", f"preds_{dataset}_{tag}.npz"))
    fp = str(d["fingerprint"])
    lo, hi = d["scale"]
    y_tr, y_te = d["y_train"], d["y_test"]
    P_tr = np.column_stack([d["oof_" + b] for b in BASE])
    P_te = np.column_stack([d["test_" + b] for b in BASE])
    inputs = {
        "preds": (P_tr, P_te),
        "features": (d["X_train"], d["X_test"]),
        "preds+features": (np.hstack([P_tr, d["X_train"]]), np.hstack([P_te, d["X_test"]])),
    }
    print(f"{dataset} {tag}  fingerprint={fp}", flush=True)

    big = len(y_tr) > 1_000_000
    for name, (inp, model) in models(big).items():
        t = time.time()
        Xtr, Xte = inputs[inp]
        if big and name == "boost_ada":
            # 20M: AdaBoost has no parallel training, so it gets 2M train ratings
            pick = np.random.default_rng(42).choice(len(y_tr), 2_000_000, replace=False)
            model.fit(Xtr[pick], y_tr[pick])
        else:
            model.fit(Xtr, y_tr)
        pred = np.clip(model.predict(Xte), lo, hi)
        sec = time.time() - t
        print(f"  {name:10s} test RMSE {rmse(pred, y_te):.4f}   ({sec:.0f} s)", flush=True)
        if name == "stack_lr":
            w = ", ".join(f"{b}={c:.2f}" for b, c in zip(BASE, model.coef_))
            print(f"             weights: {w}, intercept={model.intercept_:.2f}", flush=True)
        with open(os.path.join(RESULTS, "ensemble.jsonl"), "a") as f:
            f.write(json.dumps(dict(dataset=dataset, protocol=protocol, fold=fold, model=name,
                                    inputs=inp, fingerprint=fp, rmse=rmse(pred, y_te),
                                    mae=mae(pred, y_te), seconds=round(sec, 1))) + "\n")


if __name__ == "__main__":
    main()
