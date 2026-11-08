"""PyCaret: every regressor in one line, on our features and folds.

    python3 pycaret_compare.py mlsmall 0    # one dataset, one fold
    python3 pycaret_compare.py              # all 10
    python3 pycaret_compare.py table        # writes ../results/pycaret.md

compare_models() runs about twenty models with cross validation and gives a
table. PyCaret 3.3.2 needs Python < 3.12, so we ran it in its own environment
(Python 3.11, see requirements-pycaret.txt), not in the main one.

Same inputs as all_regressors.py: the SVR features and the same 5 outer folds
(../results/preds/preds_<ds>_kfold<f>.npz). For every outer fold:
  setup()           train rows only, 10-fold CV inside train, seed 42
  compare_models()  every model with default settings, sorted by RMSE (CV)
  predict on test   each model is refit on the whole train part by PyCaret,
                    we predict the test rows and clip to the rating scale
Left out: kernel ridge (needs an n x n matrix, 80k x 80k does not fit in
memory), SVM (one fit takes minutes, already in all_regressors.py) and
Theil-Sen (very slow). n_jobs = 2: with more, the threads of PyCaret and of the
boosting libraries add up and a laptop gets very slow. We also set
OMP_NUM_THREADS=1.

Output: ../results/pycaret.jsonl (one line per dataset, fold, model) and
PyCaret's own CV table in ../results/preds/pycaret_cv_<ds>_kfold<f>.csv
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "preds")
PATH = os.path.join(RESULTS, "pycaret.jsonl")
DATASETS = ["mlsmall", "ml100k"]
EXCLUDE = ["kr", "svm", "tr"]
N_JOBS = 2


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def done():
    if not os.path.exists(PATH):
        return set()
    return {(r["dataset"], r["fold"]) for r in map(json.loads, open(PATH))}


def run(ds, fold):
    from pycaret.regression import compare_models, predict_model, pull, setup

    z = np.load(os.path.join(OUT, f"preds_{ds}_kfold{fold}.npz"))
    cols = [f"f{i}" for i in range(z["X_train"].shape[1])]
    train = pd.DataFrame(z["X_train"], columns=cols).assign(rating=z["y_train"])
    test = pd.DataFrame(z["X_test"], columns=cols)
    y_te, scale = z["y_test"], z["scale"]

    t = time.time()
    setup(data=train, target="rating", fold=10, fold_shuffle=True, session_id=42,
          n_jobs=N_JOBS, html=False, verbose=False)
    models = compare_models(exclude=EXCLUDE, n_select=100, sort="RMSE", verbose=False)
    cv = pull()
    cv.to_csv(os.path.join(OUT, f"pycaret_cv_{ds}_kfold{fold}.csv"))
    print(f"{ds} fold {fold}: compare_models {time.time() - t:.0f} s, {len(models)} models", flush=True)

    rows = []
    # compare_models returns the models in the same order as its table (best CV RMSE first)
    for i, m in enumerate(models):
        name = cv["Model"].iloc[i]
        pred = np.clip(predict_model(m, data=test, verbose=False)["prediction_label"].to_numpy(), *scale)
        rows.append(dict(dataset=ds, fold=fold, model=name, fingerprint=str(z["fingerprint"]),
                         cv_rmse=float(cv["RMSE"].iloc[i]), rmse=rmse(pred, y_te),
                         mae=float(np.mean(np.abs(pred - y_te)))))
        print(f"  {name:32s} cv {rows[-1]['cv_rmse']:.4f}  test {rows[-1]['rmse']:.4f}", flush=True)
    with open(PATH, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def table():
    rows = list(map(json.loads, open(PATH)))
    names = sorted({r["model"] for r in rows},
                   key=lambda m: np.mean([r["rmse"] for r in rows if r["model"] == m and r["dataset"] == "mlsmall"] or [9]))
    print("| Model | Latest small: CV | Latest small: test | 100k: CV | 100k: test |")
    print("|---|---|---|---|---|")
    for m in names:
        cells = []
        for ds in DATASETS:
            v = [r for r in rows if r["model"] == m and r["dataset"] == ds]
            if v:
                cells += [f"{np.mean([r['cv_rmse'] for r in v]):.4f}",
                          f"{np.mean([r['rmse'] for r in v]):.4f} ± {np.std([r['rmse'] for r in v]):.4f} ({len(v)})"]
            else:
                cells += ["", ""]
        print(f"| {m} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    if sys.argv[1:2] == ["table"]:
        table()
    else:
        ds = [sys.argv[1]] if len(sys.argv) > 1 else DATASETS
        folds = [int(sys.argv[2])] if len(sys.argv) > 2 else range(5)
        have = done()
        for d in ds:
            for f in folds:
                if (d, f) not in have:
                    run(d, f)
        table()
