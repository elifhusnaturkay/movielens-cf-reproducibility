"""The usual RBF SVR against its fast versions: accuracy and training time.

    python3 svr_speed.py mlsmall random
    python3 svr_speed.py ml100k kfold 0

Same features and same split as the SVR in base_preds.py: they are read from
../results/preds/preds_<dataset>_<split>.npz (made by base_preds.py).
So only the model changes.

Models (all with C = 0.3, epsilon = 0.3, like the SVR in base_preds.py):
  svr_rbf       exact RBF SVR (sklearn SVR, libsvm), the model of base_preds.py
  linear_svr    LinearSVR on the features (no kernel, liblinear)
  rbfs_<n>      RBFSampler (random Fourier features, n components) + LinearSVR
  nys_<n>       Nystroem (n components) + LinearSVR

gamma is the same for all kernel models: "scale" = 1 / (n_features * X.var())
on the scaled train features, the value sklearn SVR uses.

For every model: test RMSE, MAE, train time and predict time (seconds).
Train time = transform + fit. One thread for libsvm, so we do not compare
a parallel model with a serial one. Results go to ../results/svr_speed.jsonl.

For clean times (nothing else running on the laptop):
    TIMING=1 python3 svr_speed.py mlsmall random
writes to ../results/svr_timing.jsonl instead.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
PREDS = os.path.join(RESULTS, "preds")

C, EPS = 0.3, 0.3
SIZES = [100, 300, 1000, 3000]      # number of components for the two approximations


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def linear():
    return LinearSVR(C=C, epsilon=EPS, random_state=42, max_iter=5000)


def models(gamma):
    # name -> (kernel map or None, model)
    out = {"svr_rbf": (None, SVR(C=C, epsilon=EPS, gamma=gamma, cache_size=2000)),
           "linear_svr": (None, linear())}
    for n in SIZES:
        out[f"rbfs_{n}"] = (RBFSampler(gamma=gamma, n_components=n, random_state=42), linear())
        out[f"nys_{n}"] = (Nystroem(gamma=gamma, n_components=n, random_state=42), linear())
    return out


def main():
    dataset, protocol = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    fold = int(rest.pop(0)) if protocol == "kfold" else None
    only = rest                               # optional: run only these models
    tag = protocol if fold is None else f"{protocol}{fold}"
    d = np.load(os.path.join(PREDS, f"preds_{dataset}_{tag}.npz"))
    lo, hi = d["scale"]
    sc = StandardScaler().fit(d["X_train"])
    Xtr, Xte = sc.transform(d["X_train"]), sc.transform(d["X_test"])
    ytr, yte = d["y_train"], d["y_test"]
    gamma = 1 / (Xtr.shape[1] * Xtr.var())
    fp = str(d["fingerprint"])
    print(f"{dataset} {tag}  train={len(ytr)} test={len(yte)} fingerprint={fp} "
          f"gamma={gamma:.4f}", flush=True)

    for name, (kmap, model) in models(gamma).items():
        if only and name not in only:
            continue
        warns = []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            t = time.perf_counter()
            A = kmap.fit_transform(Xtr) if kmap else Xtr
            model.fit(A, ytr)
            fit_s = time.perf_counter() - t
            t = time.perf_counter()
            B = kmap.transform(Xte) if kmap else Xte
            pred = np.clip(model.predict(B), lo, hi)
            pred_s = time.perf_counter() - t
            warns = [str(x.message)[:60] for x in w if issubclass(x.category, ConvergenceWarning)]
        extra = ""
        if name == "svr_rbf":
            # must be the same model as in base_preds.py
            same = float(np.max(np.abs(pred - d["test_svr"])))
            extra = f"  max diff to base_preds SVR {same:.1e}  support vectors {len(model.support_)}"
        print(f"  {name:11s} RMSE {rmse(pred, yte):.4f}  train {fit_s:8.1f} s  "
              f"predict {pred_s:6.2f} s{'  NOT CONVERGED' if warns else ''}{extra}", flush=True)
        out_name = "svr_timing.jsonl" if os.environ.get("TIMING") else "svr_speed.jsonl"
        with open(os.path.join(RESULTS, out_name), "a") as f:
            f.write(json.dumps(dict(dataset=dataset, protocol=protocol, fold=fold, model=name,
                                    fingerprint=fp, rmse=rmse(pred, yte), mae=mae(pred, yte),
                                    train_s=round(fit_s, 2), predict_s=round(pred_s, 2),
                                    converged=not warns)) + "\n")


if __name__ == "__main__":
    main()
