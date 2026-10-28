"""Is it the rating scale? Latest-small rounded to whole stars.

    python3 quantize.py random
    python3 quantize.py temporal
    python3 quantize.py kfold 0      (0 .. 4)

On mlsmall (half stars 0.5 .. 5) the SVR beats item KNN, on ml100k (whole
stars 1 .. 5) KNN beats the SVR. Here we make mlsmall look like ml100k:
every half star is rounded to a whole star and the models are run again.

The split is made on the original data first, so the test rows are the same
as in base_preds.py (same fingerprint). Then the rating column of train and test
is rounded, half up: 0.5->1, 1.5->2, 2.5->3, 3.5->4, 4.5->5.

Models and settings are the ones of base_preds.py (funk, knn, svr),
fit on the quantized train set, predict the quantized test set. No stacking.

Output: one line per model in ../results/quantization.jsonl. rmse_q and mae_q are
against the quantized test ratings, rmse_vs_original against the original
half star ratings. The "rounding" line is the rounding error alone.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import base_preds as bp
import common_split as cs
import eval_protocols as ep

RESULTS = os.path.join(HERE, "..", "results")

# fingerprints of the original test sets
KNOWN = {"random": "1f1d4736e82d", "temporal": "35f3d5fa32e7"}


def quantize(data):
    q = data.copy()
    q[:, 2] = np.clip(np.floor(q[:, 2] + 0.5), 1, 5)    # round half up
    return q


def main():
    protocol = sys.argv[1]
    fold = int(sys.argv[2]) if len(sys.argv) > 2 else None
    raw = cs.load_mlsmall()
    shape = (int(raw[:, 0].max()) + 1, int(raw[:, 1].max()) + 1)
    ds = {"G": ep.genres_mlsmall(), "scale": (1, 5)}

    # split first, on original ratings
    train, test = bp.get_split(raw, protocol, fold)
    fp = cs.fingerprint(test)
    print(f"mlsmall {protocol} {fold}  train={len(train)} test={len(test)} "
          f"fingerprint={fp}", flush=True)
    if protocol in KNOWN and fp != KNOWN[protocol]:
        sys.exit(f"wrong fingerprint, expected {KNOWN[protocol]}")

    # then round the ratings
    y_orig = test[:, 2].copy()
    train, test = quantize(train), quantize(test)
    y_q = test[:, 2]
    print("  train ratings now:", sorted(float(r) for r in set(train[:, 2])), flush=True)

    rows = []
    # rounding error alone
    rows.append(dict(model="rounding", rmse_q=0.0, mae_q=0.0,
                     rmse_vs_original=bp.rmse(y_q, y_orig), seconds=0.0))
    for name in ["funk", "knn", "svr"]:
        t = time.time()
        pred = bp.MODELS[name](train, test, ds, shape)
        sec = time.time() - t
        rows.append(dict(model=name, rmse_q=bp.rmse(pred, y_q), mae_q=bp.mae(pred, y_q),
                         rmse_vs_original=bp.rmse(pred, y_orig), seconds=round(sec, 1)))
        print(f"  {name:5s} RMSE(q) {rows[-1]['rmse_q']:.4f}  MAE(q) {rows[-1]['mae_q']:.4f}  "
              f"RMSE(orig) {rows[-1]['rmse_vs_original']:.4f}  ({sec:.0f} s)", flush=True)

    with open(os.path.join(RESULTS, "quantization.jsonl"), "a") as f:
        for r in rows:
            line = dict(dataset="mlsmall_q", protocol=protocol, fold=fold,
                        fingerprint_original=fp, **r)
            f.write(json.dumps(line) + "\n")
    print("  rounding error alone:", round(rows[0]["rmse_vs_original"], 4), flush=True)


if __name__ == "__main__":
    main()
