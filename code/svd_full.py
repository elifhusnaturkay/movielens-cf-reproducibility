"""Truncated SVD and Funk SVD on all splits, and Funk SVD hyperparameters.

    python3 svd_full.py protocols mlsmall      # both SVDs, all splits, K = 5, 15, 50, 100
    python3 svd_full.py hparams mlsmall        # reg_all, n_epochs, lr_all one at a time
    python3 svd_full.py hparams ml20m reg      # 20M: only reg_all, random 80/20
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import common_split as cs  # noqa: E402
from k_sweep import DATASETS, r2, rmse, tsvd_sweep  # noqa: E402
from own_models import OwnSVD  # noqa: E402

OUT = os.path.join(HERE, "..", "results")
KS = [5, 15, 50, 100]
PROTOCOLS = ([("random", None), ("temporal", None)] + [("kfold", f) for f in range(5)]
             + [("loo_last", None), ("loo_random", None)])
DEFAULTS = dict(n_epochs=20, lr=0.005, reg=0.02)
SWEEP = {"reg": [0.0, 0.01, 0.02, 0.05, 0.1, 0.2],
         "n_epochs": [5, 10, 20, 50, 100],
         "lr": [0.001, 0.002, 0.005, 0.01, 0.02]}


def get_split(raw, protocol, fold):
    if protocol == "random":
        return cs.split(raw)
    if protocol == "temporal":
        return cs.temporal_split(raw)
    if protocol == "kfold":
        return list(cs.kfold(raw, 5))[fold]
    if protocol == "loo_last":
        return cs.leave_one_out(raw, last=True)
    return cs.leave_one_out(raw, last=False)


def done(path, keys):
    if not os.path.exists(path):
        return set()
    return {tuple(json.loads(line)[k] for k in keys) for line in open(path)}


def scores(pred, y):
    return dict(rmse=rmse(pred, y), mae=float(np.mean(np.abs(pred - y))), r2=r2(pred, y))


def protocols(dataset, models):
    path = os.path.join(OUT, "svd_protocols.jsonl")
    have = done(path, ["dataset", "model", "protocol", "fold", "k"])
    ds = DATASETS[dataset]
    raw = ds["load"]()
    for protocol, fold in PROTOCOLS:
        todo = [(m, k) for m in models for k in KS
                if (dataset, m, protocol, fold, k) not in have]
        if not todo:
            continue
        train, test = get_split(raw, protocol, fold)
        y = test[:, 2]
        base = dict(dataset=dataset, protocol=protocol, fold=fold,
                    fingerprint=cs.fingerprint(test), n_test=len(test))
        print(f"[split] {dataset} {protocol} {fold} fingerprint={base['fingerprint']}", flush=True)
        rows = []
        if any(m == "tsvd" for m, _ in todo):
            for k, pred in tsvd_sweep(train, test, ds["scale"])[0]:
                if k in KS:
                    rows.append(dict(base, model="tsvd", k=k, **scores(pred, y)))
        for m, k in todo:
            if m == "funk":
                t = time.time()
                pred = OwnSVD(n_factors=k, rating_scale=ds["scale"], random_state=42) \
                    .fit(train).predict(test)
                rows.append(dict(base, model="funk", k=k, fit_s=round(time.time() - t, 1),
                                 **scores(pred, y)))
        with open(path, "a") as f:
            for r in rows:
                if (dataset, r["model"], protocol, fold, r["k"]) in have:
                    continue
                print(f"  {r['model']} K={r['k']:<3} RMSE={r['rmse']:.4f}", flush=True)
                f.write(json.dumps(r) + "\n")


def hparams(dataset, params):
    path = os.path.join(OUT, "svd_hparams.jsonl")
    have = done(path, ["dataset", "protocol", "fold", "param", "value"])
    ds = DATASETS[dataset]
    raw = ds["load"]()
    splits = [("random", None)] if dataset == "ml20m" else [("kfold", f) for f in range(5)]
    for protocol, fold in splits:
        train, test = get_split(raw, protocol, fold)
        y = test[:, 2]
        fp = cs.fingerprint(test)
        print(f"[split] {dataset} {protocol} {fold} fingerprint={fp}", flush=True)
        for param in params:
            for value in SWEEP[param]:
                if (dataset, protocol, fold, param, value) in have:
                    continue
                setting = dict(DEFAULTS, **{param: value})
                t = time.time()
                model = OwnSVD(n_factors=100, rating_scale=ds["scale"], random_state=42,
                               **setting).fit(train)
                pred = model.predict(test)
                row = dict(dataset=dataset, protocol=protocol, fold=fold, fingerprint=fp,
                           param=param, value=value, **scores(pred, y),
                           train_rmse=rmse(model.predict(train[:1_000_000]),
                                           train[:1_000_000, 2]),
                           fit_s=round(time.time() - t, 1))
                print(f"  {param}={value:<6} RMSE={row['rmse']:.4f} "
                      f"train={row['train_rmse']:.4f}", flush=True)
                with open(path, "a") as f:
                    f.write(json.dumps(row) + "\n")


def main():
    mode, dataset, *rest = sys.argv[1:]
    if mode == "protocols":
        protocols(dataset, rest or ["tsvd", "funk"])
    else:
        hparams(dataset, rest or list(SWEEP))


if __name__ == "__main__":
    main()
