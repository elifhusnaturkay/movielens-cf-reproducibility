"""RMSE for K = 0..100, truncated SVD and Funk SVD.

    python3 k_sweep.py mlsmall tsvd kfold
    python3 k_sweep.py ml20m   funk random

tsvd: scipy svds on the ratings minus the baseline (mu_u + mu_i - mu), missing = 0.
      One svds call with K = 100; smaller K are its first K columns.
funk: own_models.OwnSVD with n_factors = K, other settings at library defaults.
K = 0 means baseline / biases only. Results go to results/k_sweep.jsonl.
"""

import json
import os
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

HERE = os.path.dirname(os.path.abspath(__file__))
import common_split as cs  # noqa: E402

OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
JSONL = os.path.join(OUT, "k_sweep.jsonl")

KS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100]

DATASETS = {
    "mlsmall": dict(load=cs.load_mlsmall, scale=(0.5, 5)),
    "ml100k": dict(load=cs.load_ml100k_timed, scale=(1, 5)),
    "ml20m": dict(load=cs.load_ml20m, scale=(0.5, 5)),
}


def rmse(pred, y):
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def r2(pred, y):
    return float(1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2))


# truncated SVD
def tsvd_sweep(train, test, scale):
    # ids -> row / column numbers
    users = np.unique(np.r_[train[:, 0], test[:, 0]])
    items = np.unique(np.r_[train[:, 1], test[:, 1]])
    u_tr, i_tr = np.searchsorted(users, train[:, 0]), np.searchsorted(items, train[:, 1])
    u_te, i_te = np.searchsorted(users, test[:, 0]), np.searchsorted(items, test[:, 1])
    m, n = len(users), len(items)

    # baseline: mu_u + mu_i - mu  (unknown user or item -> mu)
    r = train[:, 2]
    mu = r.mean()
    cnt_u, cnt_i = np.bincount(u_tr, minlength=m), np.bincount(i_tr, minlength=n)
    bu = np.where(cnt_u > 0, np.bincount(u_tr, r, m) / np.maximum(cnt_u, 1) - mu, 0)
    bi = np.where(cnt_i > 0, np.bincount(i_tr, r, n) / np.maximum(cnt_i, 1) - mu, 0)

    # residual matrix, missing = 0
    A = csr_matrix((r - (mu + bu[u_tr] + bi[i_tr]), (u_tr, i_tr)), shape=(m, n))

    kmax = min(max(KS), min(m, n) - 1)
    t = time.time()
    U, s, Vt = svds(A, k=kmax, random_state=42)
    order = np.argsort(s)[::-1]               # svds gives small -> big
    U, s, Vt = U[:, order], s[order], Vt[order]
    print(f"  svds k={kmax} took {time.time() - t:.1f}s", flush=True)

    base = mu + bu[u_te] + bi[i_te]
    # contribution of each factor f to each test rating
    parts = U[u_te] * s * Vt.T[i_te]          # (n_test, kmax)
    cum = np.cumsum(parts, axis=1)
    out = []
    for k in KS:
        if k > kmax:
            continue
        pred = base if k == 0 else base + cum[:, k - 1]
        out.append((k, np.clip(pred, *scale)))
    return out, s


# Funk SVD
def funk_sweep(train, test, scale):
    from own_models import OwnSVD
    out = []
    for k in KS:
        t = time.time()
        pred = OwnSVD(n_factors=k, rating_scale=scale, random_state=42).fit(train).predict(test)
        print(f"  K={k} took {time.time() - t:.1f}s", flush=True)
        out.append((k, pred))
    return out, None


def splits(raw, protocol):
    if protocol == "random":
        yield None, cs.split(raw)
    elif protocol == "kfold":
        for f, (tr, te) in enumerate(cs.kfold(raw, 5)):
            yield f, (tr, te)
    else:
        sys.exit(f"unknown protocol {protocol}")


def main():
    dataset, model, protocol = sys.argv[1:4]
    ds = DATASETS[dataset]
    raw = ds["load"]()
    sweep = tsvd_sweep if model == "tsvd" else funk_sweep

    for fold, (train, test) in splits(raw, protocol):
        fp = cs.fingerprint(test)
        print(f"[split] k_sweep.py  {dataset}  {model}  {protocol}"
              f"{'' if fold is None else f' fold={fold}'}  train={len(train)}  "
              f"test={len(test)}  fingerprint={fp}", flush=True)
        results, s = sweep(train, test, ds["scale"])
        y = test[:, 2]
        with open(JSONL, "a") as f:
            for k, pred in results:
                row = dict(dataset=dataset, model=model, protocol=protocol, fold=fold,
                           k=k, fingerprint=fp, n_test=len(test),
                           rmse=rmse(pred, y), mae=float(np.mean(np.abs(pred - y))),
                           r2=r2(pred, y))
                print(f"  K={k:<3} RMSE={row['rmse']:.4f}", flush=True)
                f.write(json.dumps(row) + "\n")
            if s is not None and fold in (None, 0):
                f.write(json.dumps(dict(dataset=dataset, model="singular_values",
                                        protocol=protocol, s=s.tolist())) + "\n")


if __name__ == "__main__":
    main()
