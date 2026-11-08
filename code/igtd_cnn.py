"""A ready CNN from Keras Applications on the tabular features: IGTD + EfficientNetV2B0.

    python3 igtd_cnn.py                 # both datasets, 5 folds
    python3 igtd_cnn.py mlsmall 0       # one dataset, one fold
    python3 igtd_cnn.py table

A CNN needs an image, so every row of the SVR features is first turned into a
small image with IGTD (Zhu et al. 2021, Scientific Reports 11): every feature
gets one pixel, and features that are similar (high correlation) get pixels
that are close to each other. Then EfficientNetV2B0 with ImageNet weights
(Keras Applications) is fine-tuned to predict the rating.

IGTD is written here in a short form, following Section "IGTD algorithm" of the
paper: rank the feature distances (1 - |Spearman correlation|), rank the pixel
distances, and swap features between pixels while the difference between the
two rankings gets smaller. Only the train rows of the fold are used for it.

27 features -> 6 x 5 image (3 empty pixels), made 32 x 32 (the smallest input of
EfficientNetV2B0) by repeating every pixel, 3 equal color channels.
Keras runs on the PyTorch backend. One fold takes 13 to 21 minutes on a T4 GPU.

Output: ../results/igtd_cnn.jsonl, the layout in ../results/preds/igtd_layout_<ds>_kfold<f>.json
"""

import json
import os
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
OUT = os.environ.get("ENSEMBLE_OUT", os.path.join(RESULTS, "preds"))
PATH = os.path.join(os.environ.get("ENSEMBLE_OUT", RESULTS), "igtd_cnn.jsonl")
DATASETS = ["mlsmall", "ml100k"]
SIZE = 32             # image side after repeating pixels
EPOCHS = 15           # at most; early stopping on 10% of train
BATCH = 256
DEVICE = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4             # small, because the network starts from ImageNet weights


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


# ---- IGTD: features -> pixel positions ---------------------------------------
def rank_matrix(d):
    """ranks of the upper triangle of a distance matrix, as a full matrix"""
    iu = np.triu_indices(len(d), 1)
    r = np.zeros_like(d)
    r[iu] = np.argsort(np.argsort(d[iu]))
    return r + r.T


def igtd(X, rows, cols, iters=200, seed=42):
    n = X.shape[1]
    sample = X[np.random.default_rng(seed).choice(len(X), min(20000, len(X)), replace=False)]
    corr = spearmanr(sample).statistic
    corr = np.nan_to_num(corr)                      # a constant column gives nan
    feat = rank_matrix(1 - np.abs(corr))
    # pixels: grid positions; the first n of them get a feature, the rest stay empty
    pos = np.array([(i, j) for i in range(rows) for j in range(cols)], dtype=float)
    pix_all = rank_matrix(np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1)))
    order = np.arange(rows * cols)                  # order[k] = pixel of feature k (k < n)

    def error(o):
        p = pix_all[np.ix_(o[:n], o[:n])]
        return np.abs(feat - rank_matrix(p.astype(float))).sum()

    best = error(order)
    for _ in range(iters):
        improved = False
        for a in range(n):
            for b in range(rows * cols):
                if a == b:
                    continue
                o = order.copy()
                o[a], o[b] = o[b], o[a]
                e = error(o)
                if e < best:
                    best, order, improved = e, o, True
        if not improved:
            break
    return order[:n].tolist(), float(best)


def to_images(X, layout, lo, hi, rows, cols):
    Z = np.clip((X - lo) / np.where(hi > lo, hi - lo, 1), 0, 1)
    img = np.zeros((len(X), rows * cols), dtype="float32")
    img[:, layout] = Z
    img = img.reshape(len(X), rows, cols)
    rep_r, rep_c = -(-SIZE // rows), -(-SIZE // cols)           # ceil
    img = img.repeat(rep_r, 1).repeat(rep_c, 2)[:, :SIZE, :SIZE]
    return np.repeat(img[..., None] * 255.0, 3, axis=-1)       # model expects 0-255


# ---- the CNN from Keras Applications ------------------------------------------
def make_model(scale):
    base = keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet",
                                               input_shape=(SIZE, SIZE, 3), pooling="avg")
    x = keras.layers.Dropout(0.2)(base.output)
    x = keras.layers.Dense(1)(x)
    # output in the rating range: mean + scaled tanh would also work; we just clip later
    model = keras.Model(base.input, x)
    model.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")
    return model


def done():
    if not os.path.exists(PATH):
        return set()
    return {(r["dataset"], r["fold"]) for r in map(json.loads, open(PATH))}


def run(datasets, folds):
    have = done()
    keras.utils.set_random_seed(42)
    for ds in datasets:
        for fold in folds:
            if (ds, fold) in have:
                continue
            z = np.load(os.path.join(OUT, f"preds_{ds}_kfold{fold}.npz"))
            Xtr, Xte, ytr, yte, scale = z["X_train"], z["X_test"], z["y_train"], z["y_test"], z["scale"]
            n = Xtr.shape[1]
            rows = int(np.ceil(np.sqrt(n)))
            cols = int(np.ceil(n / rows))
            t = time.time()
            layout, err = igtd(Xtr, rows, cols)
            with open(os.path.join(OUT, f"igtd_layout_{ds}_kfold{fold}.json"), "w") as f:
                json.dump(dict(rows=rows, cols=cols, layout=layout, error=err), f)
            lo, hi = np.percentile(Xtr, 1, axis=0), np.percentile(Xtr, 99, axis=0)
            Itr, Ite = to_images(Xtr, layout, lo, hi, rows, cols), to_images(Xte, layout, lo, hi, rows, cols)
            mean = float(ytr.mean())
            model = make_model(scale)
            stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            hist = model.fit(Itr, ytr - mean, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH,
                             callbacks=[stop], verbose=2)
            pred = np.clip(model.predict(Ite, batch_size=1024, verbose=0).ravel() + mean, *scale)
            row = dict(dataset=ds, fold=fold, model="IGTD + EfficientNetV2B0", image=f"{rows}x{cols}->{SIZE}",
                       fingerprint=str(z["fingerprint"]), device=DEVICE, rmse=rmse(pred, yte),
                       mae=float(np.mean(np.abs(pred - yte))), epochs=len(hist.history["loss"]),
                       seconds=round(time.time() - t, 1))
            with open(PATH, "a") as f:
                f.write(json.dumps(row) + "\n")
            np.save(os.path.join(OUT, f"igtd_cnn_{ds}_kfold{fold}.npy"), pred)
            print(f"{ds} fold {fold}  RMSE {row['rmse']:.4f}  epochs {row['epochs']}  {row['seconds']}s", flush=True)


def table():
    rows = list(map(json.loads, open(PATH)))
    for ds in DATASETS:
        v = [r["rmse"] for r in rows if r["dataset"] == ds]
        if v:
            print(f"{ds:8s} folds {len(v)}  RMSE {np.mean(v):.4f} ± {np.std(v):.4f}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["table"]:
        table()
    else:
        ds = [sys.argv[1]] if len(sys.argv) > 1 else DATASETS
        folds = [int(sys.argv[2])] if len(sys.argv) > 2 else range(5)
        run(ds, folds)
        table()
