"""Ensemble of the deep learning models by simple averaging.

    python3 dl_average.py

No weights are learned, every model gets the same weight, so the test set is
not used for any choice. Inputs (test predictions in ../results/preds, same
order as y_test):
  ncf, mlp, hybrid      save_dl_preds.py, seed 42
  gru, lstm             rnn.py, seeds 42 43 44
  bst, bert             bst.py, seeds 42 43 44
Results go to ../results/dl_average.jsonl.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
PREDS = os.path.join(RESULTS, "preds")

WHERE = {m: PREDS for m in ["ncf", "mlp", "hybrid", "gru", "lstm", "bst", "bert"]}
SEEDS = {"ncf": [42], "mlp": [42], "hybrid": [42],
         "gru": [42, 43, 44], "lstm": [42, 43, 44], "bst": [42, 43, 44], "bert": [42, 43, 44]}

# which models go into each average
GROUPS = {
    "avg_transformers": ["bst", "bert"],
    "avg_rnn": ["gru", "lstm"],
    "avg_sequence": ["gru", "lstm", "bst", "bert"],
    "avg_nonseq": ["ncf", "mlp", "hybrid"],
    "avg_one_per_family": ["hybrid", "gru", "bert"],
    "avg_all_dl": ["ncf", "mlp", "hybrid", "gru", "lstm", "bst", "bert"],
}


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def stack_xgb(dataset, protocol, fold):
    for line in open(os.path.join(RESULTS, "ensemble.jsonl")):
        r = json.loads(line)
        if (r["dataset"], r["protocol"], r.get("fold"), r["model"]) == (dataset, protocol, fold, "stack_xgb"):
            best = r["rmse"]
    return best


def run(dataset, protocol, fold):
    tag = protocol if fold is None else f"{protocol}{fold}"
    d = np.load(os.path.join(PREDS, f"preds_{dataset}_{tag}.npz"))
    y, lo, hi = d["y_test"], d["scale"][0], d["scale"][1]
    rows = []

    # one prediction per model: the mean over its seeds
    pred = {}
    for m in WHERE:
        ps = [np.load(os.path.join(WHERE[m], f"pred_{dataset}_{tag}_{m}_{s}.npy")) for s in SEEDS[m]]
        pred[m] = np.mean(ps, axis=0)
        rows.append(dict(model=m, kind="single", rmse=float(np.mean([rmse(p, y) for p in ps]))))
        if len(ps) > 1:
            rows.append(dict(model=m + "_seedavg", kind="seed average", rmse=rmse(pred[m], y)))

    for name, ms in GROUPS.items():
        p = np.clip(np.mean([pred[m] for m in ms], axis=0), lo, hi)
        rows.append(dict(model=name, kind="average", members=ms, rmse=rmse(p, y)))

    rows.append(dict(model="stack_xgb", kind="reference (stacking)", rmse=stack_xgb(dataset, protocol, fold)))
    for r in rows:
        r.update(dataset=dataset, protocol=protocol, fold=fold, fingerprint=str(d["fingerprint"]))
    return rows


if __name__ == "__main__":
    out = os.path.join(RESULTS, "dl_average.jsonl")
    open(out, "w").close()
    for ds in ("mlsmall", "ml100k"):
        splits = [("random", None), ("temporal", None)] + [("kfold", k) for k in range(5)]
        for protocol, fold in splits:
            with open(out, "a") as f:
                for r in run(ds, protocol, fold):
                    f.write(json.dumps(r) + "\n")
    print("wrote", out)
