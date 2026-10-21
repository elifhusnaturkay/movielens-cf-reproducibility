"""Stacking of Funk SVD, co-clustering, KNN and SVR.

    python3 stack.py mlsmall random
    python3 stack.py ml100k kfold 0

Reads ../results/preds/preds_<dataset>_<split>.npz (made by base_preds.py).
The meta model learns from the out-of-fold predictions only.

  stack_lr    linear regression on [funk, cc, knn, svr]
  stack_xgb   XGBoost on [funk, cc, knn, svr] + the features
              (400 trees, depth 6, learning rate 0.05, not tuned)

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


def main():
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor

    dataset, protocol = sys.argv[1], sys.argv[2]
    fold = int(sys.argv[3]) if len(sys.argv) > 3 else None
    tag = protocol if fold is None else f"{protocol}{fold}"
    d = np.load(os.path.join(RESULTS, "preds", f"preds_{dataset}_{tag}.npz"))
    fp = str(d["fingerprint"])
    lo, hi = d["scale"]
    y_tr, y_te = d["y_train"], d["y_test"]
    P_tr = np.column_stack([d["oof_" + b] for b in BASE])
    P_te = np.column_stack([d["test_" + b] for b in BASE])
    models = {
        "stack_lr": (LinearRegression(), P_tr, P_te),
        "stack_xgb": (XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42,
                                   n_jobs=8),
                      np.hstack([P_tr, d["X_train"]]), np.hstack([P_te, d["X_test"]])),
    }
    print(f"{dataset} {tag}  fingerprint={fp}", flush=True)
    for name, (model, Xtr, Xte) in models.items():
        t = time.time()
        model.fit(Xtr, y_tr)
        pred = np.clip(model.predict(Xte), lo, hi)
        print(f"  {name:10s} test RMSE {rmse(pred, y_te):.4f}", flush=True)
        if name == "stack_lr":
            w = ", ".join(f"{b}={c:.2f}" for b, c in zip(BASE, model.coef_))
            print(f"             weights: {w}, intercept={model.intercept_:.2f}", flush=True)
        with open(os.path.join(RESULTS, "ensemble.jsonl"), "a") as f:
            f.write(json.dumps(dict(dataset=dataset, protocol=protocol, fold=fold, model=name,
                                    fingerprint=fp, rmse=rmse(pred, y_te), mae=mae(pred, y_te),
                                    seconds=round(time.time() - t, 1))) + "\n")


if __name__ == "__main__":
    main()
