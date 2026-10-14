"""Tests for the six co-clustering bases (bases.py).

    python3 run_bases.py protocols mlsmall          # 3 x 3, all splits
    python3 run_bases.py grid ml100k kfold          # 9 x 9 grid of K x L, 5 folds
    python3 run_bases.py grid ml20m random square   # K = L only
    python3 run_bases.py best ml100k                # each basis at its best K x L

Runs already in results/protocols.jsonl or results/grid.jsonl are skipped.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import common_split as cs  # noqa: E402
from bases import BASES, BasisCoClustering  # noqa: E402

OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

SIZES = [1, 2, 3, 5, 10, 15, 20, 30, 50]
DATASETS = {
    "mlsmall": dict(load=cs.load_mlsmall, scale=(0.5, 5)),
    "ml100k": dict(load=cs.load_ml100k_timed, scale=(1, 5)),
    "ml20m": dict(load=cs.load_ml20m, scale=(0.5, 5)),
}
# best cell of each basis in the grid (results/grid.jsonl)
BEST = {
    "mlsmall": {"C1": (15, 50), "C2": (30, 3), "C3": (10, 5), "C4": (30, 1),
                "C5": (50, 2), "C6": (1, 1)},
    "ml100k": {"C1": (30, 50), "C2": (15, 5), "C3": (10, 10), "C4": (50, 2),
               "C5": (15, 2), "C6": (1, 2)},
    "ml20m": {"C1": (50, 50), "C2": (30, 30), "C3": (30, 30), "C4": (50, 50),
              "C5": (50, 50), "C6": (2, 2)},
}
PROTOCOLS = ([("random", None), ("temporal", None)] + [("kfold", f) for f in range(5)]
             + [("loo_last", None), ("loo_random", None)])


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


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def done(path, keys):
    if not os.path.exists(path):
        return set()
    return {tuple(json.loads(line)[k] for k in keys) for line in open(path)}


def run_one(basis, k, l, train, test, shape, scale):
    t = time.time()
    model = BasisCoClustering(basis, k, l, max_iter=50, random_state=42).fit(train, shape)
    pred = model.predict(test)
    y = test[:, 2]
    # train error on (at most) 1M training ratings, enough to see overfitting
    sample = train if len(train) <= 1_000_000 else train[:: len(train) // 1_000_000]
    return dict(basis=basis, k=k, l=l, n_test=len(test),
                rmse=rmse(pred, y), rmse_clip=rmse(np.clip(pred, *scale), y),
                mae=float(np.mean(np.abs(pred - y))),
                r2=float(1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)),
                train_rmse=rmse(model.predict(sample), sample[:, 2]),
                n_iter=model.n_iter, fit_s=round(time.time() - t, 1))


def protocols(dataset, bases, size=(3, 3)):
    """size: (K, L) for every basis, or a dict {basis: (K, L)}."""
    path = os.path.join(OUT, "protocols.jsonl")
    have = done(path, ["dataset", "basis", "protocol", "fold", "k", "l"])
    kl = size if isinstance(size, dict) else {b: size for b in bases}
    ds = DATASETS[dataset]
    raw = ds["load"]()
    shape = (int(raw[:, 0].max()) + 1, int(raw[:, 1].max()) + 1)
    for protocol, fold in PROTOCOLS:
        todo = [b for b in bases if (dataset, b, protocol, fold, *kl[b]) not in have]
        if not todo:
            continue
        train, test = get_split(raw, protocol, fold)
        fp = cs.fingerprint(test)
        print(f"[split] run_bases.py  {dataset}  {protocol}"
              f"{'' if fold is None else f' fold={fold}'}  train={len(train)}  "
              f"test={len(test)}  fingerprint={fp}", flush=True)
        for b in todo:
            row = dict(dataset=dataset, protocol=protocol, fold=fold, fingerprint=fp,
                       **run_one(b, *kl[b], train, test, shape, ds["scale"]))
            print(f"  {b} {kl[b][0]}x{kl[b][1]}  RMSE={row['rmse']:.4f}  train={row['train_rmse']:.4f}  "
                  f"it={row['n_iter']}  {row['fit_s']}s", flush=True)
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")


def grid(dataset, protocol, square, bases):
    path = os.path.join(OUT, "grid.jsonl")
    have = done(path, ["dataset", "basis", "protocol", "fold", "k", "l"])
    ds = DATASETS[dataset]
    raw = ds["load"]()
    shape = (int(raw[:, 0].max()) + 1, int(raw[:, 1].max()) + 1)
    cells = [(k, k) for k in SIZES] if square else [(k, l) for k in SIZES for l in SIZES]
    folds = [None] if protocol == "random" else range(5)
    for fold in folds:
        train, test = get_split(raw, protocol, fold)
        fp = cs.fingerprint(test)
        print(f"[split] run_bases.py  {dataset}  {protocol}"
              f"{'' if fold is None else f' fold={fold}'}  fingerprint={fp}", flush=True)
        for b in bases:
            for k, l in cells:
                if (dataset, b, protocol, fold, k, l) in have:
                    continue
                row = dict(dataset=dataset, protocol=protocol, fold=fold, fingerprint=fp,
                           **run_one(b, k, l, train, test, shape, ds["scale"]))
                print(f"  {b} {k:>2} x {l:<2} RMSE={row['rmse']:.4f}  "
                      f"train={row['train_rmse']:.4f}  {row['fit_s']}s", flush=True)
                with open(path, "a") as f:
                    f.write(json.dumps(row) + "\n")


def main():
    args = sys.argv[1:]
    bases = [a for a in args if a in BASES] or BASES
    args = [a for a in args if a not in BASES]
    if args[0] == "protocols":
        protocols(args[1], bases)
    elif args[0] == "best":
        # every basis at its best K x L from the grid (5-fold mean; 20M: random)
        protocols(args[1], bases, BEST[args[1]])
    else:
        grid(args[1], args[2], "square" in args, bases)


if __name__ == "__main__":
    main()
