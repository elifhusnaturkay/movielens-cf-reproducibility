"""Deep learning: NCF, MLP and a hybrid of both.

    python3 dl.py mlsmall random
    python3 dl.py ml100k kfold 0

Reads ../results/preds/preds_<dataset>_<split>.npz (made by base_preds.py),
so the split and the features are exactly the same as for the other models.

  ncf     Neural Collaborative Filtering (He et al. 2017): user and movie
          embeddings, a GMF part (element-wise product) and an MLP part
          (3 hidden layers), predicts the rating. Only ids, no features.
  mlp     the same features as the SVR, 3 hidden layers (128, 64, 32).
  hybrid  ncf + the features in one network.

Training: Adam, MSE loss, batch 1024 (8192 on 20M), at most 40 epochs. 10% of the train
set is kept aside for early stopping (stop after 3 epochs without
improvement, keep the best epoch). The test set is used once, at the end.
Users or movies not in train get embedding 0 (padding_idx), so they fall
back to the bias part. Seed 42.
Results go to ../results/deep.jsonl.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
DIM = 32


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def index(ids, known):
    """raw id -> 1..n if seen in train, 0 if not."""
    pos = np.clip(np.searchsorted(known, ids), 0, len(known) - 1)
    return np.where(known[pos] == ids, pos + 1, 0)


def mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(0.2)]
    return nn.Sequential(*layers)


class Net(nn.Module):
    def __init__(self, kind, n_users, n_items, n_feat, mu):
        super().__init__()
        self.kind = kind
        self.mu = mu
        size = 0
        if kind in ("ncf", "hybrid"):
            self.ug = nn.Embedding(n_users + 1, DIM, padding_idx=0)
            self.ig = nn.Embedding(n_items + 1, DIM, padding_idx=0)
            self.um = nn.Embedding(n_users + 1, DIM, padding_idx=0)
            self.im = nn.Embedding(n_items + 1, DIM, padding_idx=0)
            self.ub = nn.Embedding(n_users + 1, 1, padding_idx=0)
            self.ib = nn.Embedding(n_items + 1, 1, padding_idx=0)
            for e in (self.ug, self.ig, self.um, self.im):
                nn.init.normal_(e.weight, 0, 0.05)
            for e in (self.ub, self.ib):
                nn.init.zeros_(e.weight)
            self.ids_mlp = mlp([2 * DIM, 64, 32, 16])
            size += DIM + 16
        if kind in ("mlp", "hybrid"):
            self.feat_mlp = mlp([n_feat, 128, 64, 32])
            size += 32
        self.out = nn.Linear(size, 1)

    def forward(self, u, i, x):
        parts = []
        bias = 0
        if self.kind in ("ncf", "hybrid"):
            parts.append(self.ug(u) * self.ig(i))
            parts.append(self.ids_mlp(torch.cat([self.um(u), self.im(i)], 1)))
            bias = self.ub(u).squeeze(1) + self.ib(i).squeeze(1)
        if self.kind in ("mlp", "hybrid"):
            parts.append(self.feat_mlp(x))
        return self.mu + bias + self.out(torch.cat(parts, 1)).squeeze(1)


def predict(net, u, i, x):
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(u), 8192):
            out.append(net(u[s:s + 8192], i[s:s + 8192], x[s:s + 8192]).cpu().numpy())
    return np.concatenate(out)


def train_one(kind, d):
    batch = 8192 if len(d["train"]) > 1_000_000 else 1024   # 20M: bigger batches
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    tr, te = d["train"], d["test"]
    users, items = np.unique(tr[:, 0]), np.unique(tr[:, 1])
    Xtr, Xte = d["X_train"], d["X_test"]
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-6           # scale with train numbers only
    Xtr, Xte = (Xtr - m) / s, (Xte - m) / s

    def tensors(rows, X):
        return (torch.tensor(index(rows[:, 0], users), device=DEV),
                torch.tensor(index(rows[:, 1], items), device=DEV),
                torch.tensor(X, dtype=torch.float32, device=DEV))

    order = rng.permutation(len(tr))
    cut = int(len(tr) * 0.9)
    fit, val = order[:cut], order[cut:]
    U, I, X = tensors(tr, Xtr)
    Y = torch.tensor(tr[:, 2], dtype=torch.float32, device=DEV)
    Ut, It, XT = tensors(te, Xte)

    net = Net(kind, len(users), len(items), Xtr.shape[1], float(tr[fit, 2].mean())).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state, wait, epochs = 9.0, None, 0, 0
    fit_t = torch.tensor(fit, device=DEV)
    y_val = tr[val, 2]
    for epoch in range(40):
        net.train()
        perm = fit_t[torch.randperm(len(fit_t), device=DEV)]
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            loss = ((net(U[b], I[b], X[b]) - Y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = torch.tensor(val, device=DEV)
        score = rmse(predict(net, U[v], I[v], X[v]), y_val)
        if score < best - 1e-4:
            best, wait, epochs = score, 0, epoch + 1
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= 3:
                break
    net.load_state_dict(best_state)
    lo, hi = d["scale"]
    return np.clip(predict(net, Ut, It, XT), lo, hi), best, epochs


def main():
    dataset, protocol = sys.argv[1], sys.argv[2]
    fold = int(sys.argv[3]) if len(sys.argv) > 3 else None
    tag = protocol if fold is None else f"{protocol}{fold}"
    d = dict(np.load(os.path.join(RESULTS, "preds", f"preds_{dataset}_{tag}.npz")))
    fp = str(d["fingerprint"])
    y_te = d["y_test"]
    print(f"{dataset} {tag}  fingerprint={fp}  device={DEV}", flush=True)
    for kind in ("ncf", "mlp", "hybrid"):
        t = time.time()
        pred, val_rmse, epochs = train_one(kind, d)
        sec = time.time() - t
        print(f"  {kind:7s} test RMSE {rmse(pred, y_te):.4f}   val {val_rmse:.4f}   "
              f"best epoch {epochs}   ({sec:.0f} s)", flush=True)
        with open(os.path.join(RESULTS, "deep.jsonl"), "a") as f:
            f.write(json.dumps(dict(dataset=dataset, protocol=protocol, fold=fold,
                                    model=kind, fingerprint=fp,
                                    rmse=rmse(pred, y_te), mae=mae(pred, y_te),
                                    val_rmse=val_rmse, best_epoch=epochs,
                                    seconds=round(sec, 1))) + "\n")


if __name__ == "__main__":
    main()
