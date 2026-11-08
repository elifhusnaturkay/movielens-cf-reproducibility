"""A tabular foundation model from Hugging Face: TabICLv2.

    python3 tabular_fm.py 20000            # context of 20,000 train rows
    python3 tabular_fm.py 20000 mlsmall 0  # one dataset, one fold
    python3 tabular_fm.py table

TabICLv2 (Qu et al., ICML 2026, soda-inria/tabicl) was pretrained on millions of
synthetic tables. It is NOT trained on our data: it reads train rows as
"context" and predicts the test rows in one forward pass (in-context learning).

Input: the SVR features (27 columns on latest-small, 26 on ml-100k) and the
same 5 folds, from ../results/preds/preds_<ds>_kfold<f>.npz (base_preds.py).
The context is a random sample of the train rows (seed 42). The whole train
set (80,000 rows) did not fit in 15 GB of GPU memory.
Set ENSEMBLE_OUT to read and write another folder (we used it on Colab).

Output: ../results/tabular_fm.jsonl
"""

import json
import os
import sys
import time

import numpy as np
import torch
from tabicl import TabICLRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
OUT = os.environ.get("ENSEMBLE_OUT", os.path.join(RESULTS, "preds"))
PATH = os.path.join(os.environ.get("ENSEMBLE_OUT", RESULTS), "tabular_fm.jsonl")
DATASETS = ["mlsmall", "ml100k"]
DEVICE = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CHUNK = 2000          # test rows per predict call, so the GPU memory is enough


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def done():
    if not os.path.exists(PATH):
        return set()
    return {(r["dataset"], r["fold"], r["context"]) for r in map(json.loads, open(PATH))}


def run(context, datasets, folds):
    have = done()
    for ds in datasets:
        for fold in folds:
            if (ds, fold, context) in have:
                continue
            z = np.load(os.path.join(OUT, f"preds_{ds}_kfold{fold}.npz"))
            Xtr, Xte, ytr, yte, scale = z["X_train"], z["X_test"], z["y_train"], z["y_test"], z["scale"]
            idx = np.random.default_rng(42).choice(len(Xtr), min(context, len(Xtr)), replace=False)
            t = time.time()
            model = TabICLRegressor(random_state=42, batch_size=1, kv_cache=True)
            model.fit(Xtr[idx], ytr[idx])
            pred = np.concatenate([model.predict(Xte[i:i + CHUNK]) for i in range(0, len(Xte), CHUNK)])
            pred = np.clip(pred, *scale)
            row = dict(dataset=ds, fold=fold, model="TabICLv2", context=int(len(idx)),
                       fingerprint=str(z["fingerprint"]), device=DEVICE, rmse=rmse(pred, yte),
                       mae=float(np.mean(np.abs(pred - yte))), seconds=round(time.time() - t, 1))
            with open(PATH, "a") as f:
                f.write(json.dumps(row) + "\n")
            np.save(os.path.join(OUT, f"tabicl_{ds}_kfold{fold}_ctx{len(idx)}.npy"), pred)
            print(f"{ds} fold {fold} context {len(idx)}  RMSE {row['rmse']:.4f}  {row['seconds']}s", flush=True)


def table():
    rows = list(map(json.loads, open(PATH)))
    for ds in DATASETS:
        for c in sorted({r["context"] for r in rows}):
            v = [r["rmse"] for r in rows if r["dataset"] == ds and r["context"] == c]
            if v:
                print(f"{ds:8s} context {c:6d}  folds {len(v)}  RMSE {np.mean(v):.4f} ± {np.std(v):.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "table":
        table()
    else:
        ctx = int(sys.argv[1])
        ds = [sys.argv[2]] if len(sys.argv) > 2 else DATASETS
        folds = [int(sys.argv[3])] if len(sys.argv) > 3 else range(5)
        run(ctx, ds, folds)
        table()
