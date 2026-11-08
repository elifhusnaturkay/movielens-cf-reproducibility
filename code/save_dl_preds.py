"""Save the test predictions of NCF, MLP and hybrid (dl.py does not save them).

    python3 save_dl_preds.py mlsmall random
    python3 save_dl_preds.py ml100k kfold 3

Same train_one as dl.py (seed 42), so the numbers match ../results/deep.jsonl
up to GPU noise.
"""

import os
import sys

import numpy as np

from dl import train_one, rmse

HERE = os.path.dirname(os.path.abspath(__file__))
PREDS = os.path.join(HERE, "..", "results", "preds")

dataset, protocol = sys.argv[1], sys.argv[2]
fold = int(sys.argv[3]) if len(sys.argv) > 3 else None
tag = protocol if fold is None else f"{protocol}{fold}"
d = dict(np.load(os.path.join(PREDS, f"preds_{dataset}_{tag}.npz")))
for kind in ("ncf", "mlp", "hybrid"):
    pred, val, ep = train_one(kind, d)
    print(f"{dataset} {tag} {kind}: test {rmse(pred, d['y_test']):.4f}", flush=True)
    np.save(os.path.join(PREDS, f"pred_{dataset}_{tag}_{kind}_42.npy"), pred)
