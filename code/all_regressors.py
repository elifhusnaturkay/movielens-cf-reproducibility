"""Many regression models at once, all with their default settings.

    python3 all_regressors.py          # runs what is missing, then the table
    python3 all_regressors.py table    # only the table

About 18 regressors of scikit-learn, XGBoost and CatBoost, all with their
DEFAULT settings, on the SVR features (features.py, out-of-fold for the train
rows, saved by base_preds.py). Latest-small and ml-100k, the same 5 folds.

This table answers: with only the features (no collaborative model), which
library model is best, and are our hand-set SVR, Random Forest and XGBoost
close? Stacking uses the collaborative predictions too, so it is not in this race.

Output: ../results/all_regressors.jsonl and ../results/all_regressors.md
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import (BayesianRidge, ElasticNet, HuberRegressor, Lasso,
                                  LinearRegression, Ridge)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "preds")
PATH = os.path.join(RESULTS, "all_regressors.jsonl")
DATASETS = ["mlsmall", "ml100k"]
JOBS = 2          # few cores, so the computer stays cool


def scaled(model):
    return make_pipeline(StandardScaler(), model)


MODELS = {
    "Mean (baseline)": lambda: DummyRegressor(),
    "Linear regression": lambda: scaled(LinearRegression()),
    "Ridge": lambda: scaled(Ridge(random_state=42)),
    "Lasso": lambda: scaled(Lasso(random_state=42)),
    "Elastic net": lambda: scaled(ElasticNet(random_state=42)),
    "Bayesian ridge": lambda: scaled(BayesianRidge()),
    "Huber": lambda: scaled(HuberRegressor(max_iter=500)),
    "KNN regressor": lambda: scaled(KNeighborsRegressor(n_jobs=JOBS)),
    "Decision tree": lambda: DecisionTreeRegressor(random_state=42),
    "Random forest": lambda: RandomForestRegressor(random_state=42, n_jobs=JOBS),
    "Extra trees": lambda: ExtraTreesRegressor(random_state=42, n_jobs=JOBS),
    "AdaBoost": lambda: AdaBoostRegressor(random_state=42),
    "Gradient boosting": lambda: GradientBoostingRegressor(random_state=42),
    "Hist gradient boosting": lambda: HistGradientBoostingRegressor(random_state=42),
    "XGBoost": lambda: XGBRegressor(random_state=42, n_jobs=JOBS),
    "CatBoost": lambda: CatBoostRegressor(random_seed=42, verbose=0, thread_count=JOBS,
                                          train_dir=os.path.join(RESULTS, "preds", "catboost_info")),
    "MLP": lambda: scaled(MLPRegressor(random_state=42)),
    "SVR (RBF)": lambda: scaled(SVR()),
}

# our own hand-set models (../results/ensemble.jsonl and deep.jsonl), for the table
OURS = {"svr": "our SVR (C = 0.3, ε = 0.3)", "bag_rf": "our Random Forest (300 trees)",
        "boost_xgb": "our XGBoost (400 trees, depth 6)", "boost_cat": "our CatBoost",
        "boost_ada": "our AdaBoost", "knn": "KNN, item based",
        "stack_xgb": "stacking with XGBoost (uses the collaborative models)"}


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def done():
    if not os.path.exists(PATH):
        return set()
    return {(r["dataset"], r["fold"], r["model"]) for r in map(json.loads, open(PATH))}


def run():
    have = done()
    for ds in DATASETS:
        for fold in range(5):
            z = np.load(os.path.join(OUT, f"preds_{ds}_kfold{fold}.npz"))
            Xtr, Xte, ytr, yte, scale = z["X_train"], z["X_test"], z["y_train"], z["y_test"], z["scale"]
            for name, make in MODELS.items():
                if (ds, fold, name) in have:
                    continue
                t = time.time()
                pred = np.clip(make().fit(Xtr, ytr).predict(Xte), *scale)
                row = dict(dataset=ds, fold=fold, model=name, fingerprint=str(z["fingerprint"]),
                           rmse=rmse(pred, yte), mae=float(np.mean(np.abs(pred - yte))),
                           seconds=round(time.time() - t, 1))
                with open(PATH, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{ds} fold {fold}  {name:24s} RMSE {row['rmse']:.4f}  {row['seconds']}s", flush=True)


def table():
    rows = list(map(json.loads, open(PATH)))
    ours = [r for r in map(json.loads, open(os.path.join(RESULTS, "ensemble.jsonl"))) if r["protocol"] == "kfold"]
    lines = ["# Every regressor with default settings, on the SVR features", "",
             "5-fold mean RMSE (± std over the folds). Sorted by latest small. Clipped to the rating scale.", "",
             "| Model | Latest small | 100k | Time per fold (s, latest small) |", "|---|---|---|---|"]

    def cell(rs):
        v = [r["rmse"] for r in rs]
        return f"{np.mean(v):.4f} ± {np.std(v):.4f}" if len(v) == 5 else "–"

    order = sorted(MODELS, key=lambda m: np.mean([r["rmse"] for r in rows if r["model"] == m
                                                  and r["dataset"] == "mlsmall"] or [9]))
    for m in order:
        cells = [cell([r for r in rows if r["model"] == m and r["dataset"] == d]) for d in DATASETS]
        sec = [r["seconds"] for r in rows if r["model"] == m and r["dataset"] == "mlsmall"]
        lines.append(f"| {m} | {cells[0]} | {cells[1]} | {np.mean(sec):.0f} |" if sec else f"| {m} | – | – | – |")
    lines += ["", "Our own hand-set models on the same folds (from ensemble.jsonl):", "",
              "| Model | Latest small | 100k |", "|---|---|---|"]
    for key, name in OURS.items():
        cells = [cell([r for r in ours if r["model"] == key and r["dataset"] == d]) for d in DATASETS]
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
    with open(os.path.join(RESULTS, "all_regressors.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if "table" not in sys.argv:
        run()
    table()
