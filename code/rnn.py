"""GRU and LSTM over the user's rating sequence.

    python3 rnn.py mlsmall temporal gru 42
    python3 rnn.py ml100k kfold 0 lstm 43

Same data pipeline as bst.py (index, histories and get_split are imported
from there), so the numbers can be compared directly:
the last L=20 train ratings of the user before the target, each token =
movie embedding + rating embedding, the last token is the target movie with
a [MASK] rating. Only the encoder is different: the 21 tokens go through a
GRU or an LSTM (2 layers, hidden 64) in time order, and we take the output
at the last position (the target token). Padding is on the left, so the
last position is always the target. No position embedding, the RNN already
reads the tokens in order.

Prediction = global mean + user bias + movie bias + MLP([h, user embedding]).
No leakage: histories come from the train set only (assert kept).
Early stopping on 10% of train. Results go to ../results/rnn.jsonl, test
predictions to ../results/preds/.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
import common_split as cs
from bst import DATASETS, L, D, get_split, histories, index, rmse, mae

RESULTS = os.path.join(HERE, "..", "results")
PREDS = os.path.join(RESULTS, "preds")
os.makedirs(PREDS, exist_ok=True)
DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, kind, n_users, n_items, n_levels, mu):
        super().__init__()
        self.mu = mu
        self.item = nn.Embedding(n_items + 1, D, padding_idx=0)
        self.rate = nn.Embedding(n_levels + 2, D)            # levels, [MASK], [PAD]
        self.user = nn.Embedding(n_users + 1, D, padding_idx=0)
        self.ub = nn.Embedding(n_users + 1, 1, padding_idx=0)
        self.ib = nn.Embedding(n_items + 1, 1, padding_idx=0)
        nn.init.zeros_(self.ub.weight)
        nn.init.zeros_(self.ib.weight)
        rnn = nn.GRU if kind == "gru" else nn.LSTM
        self.enc = rnn(D, 64, num_layers=2, dropout=0.1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64 + D, 64), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(64, 1))

    def forward(self, u, hist_i, hist_r, target):
        # history tokens, then target token
        items = torch.cat([hist_i, target[:, None]], 1)
        mask_r = torch.full_like(target, self.rate.num_embeddings - 2)
        rates = torch.cat([hist_r, mask_r[:, None]], 1)
        x = self.item(items) + self.rate(rates)
        h = self.enc(x)[0][:, -1]                            # last step = target
        out = self.head(torch.cat([h, self.user(u)], 1)).squeeze(1)
        return self.mu + self.ub(u).squeeze(1) + self.ib(target).squeeze(1) + out


def main():
    dataset, protocol = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    fold = int(rest.pop(0)) if protocol == "kfold" else None
    kind, seed = rest[0], int(rest[1])
    if kind not in ("gru", "lstm"):
        sys.exit(f"unknown model {kind}")
    ds = DATASETS[dataset]
    raw = ds["load"]()
    train, test = get_split(raw, protocol, fold)
    fp = cs.fingerprint(test)
    lo, hi = ds["scale"]

    users, items = np.unique(train[:, 0]), np.unique(train[:, 1])
    levels = np.unique(raw[:, 2])                            # 10 half stars or 5 stars
    r_tok = np.searchsorted(levels, train[:, 2])             # rating -> 0..n_levels-1
    pad_r = len(levels) + 1

    Htr = histories(train, train, True)
    Hte = histories(train, test, False)
    # leakage check
    ok = Hte >= 0
    rows = np.nonzero(ok.any(1))[0][:2000]
    for k in rows:
        h = Hte[k][Hte[k] >= 0]
        assert (train[h, 0] == test[k, 0]).all() and (train[h, 3] < test[k, 3]).all()
    print(f"{dataset} {protocol} {fold} {kind} seed={seed}  train={len(train)} "
          f"test={len(test)} fingerprint={fp}  device={DEV}  "
          f"mean history: train {(Htr >= 0).sum(1).mean():.1f}, test {ok.sum(1).mean():.1f}",
          flush=True)

    def tensors(rows, H):
        hi_ = np.empty(H.shape, np.int32)
        hr_ = np.empty(H.shape, np.int32)
        for s in range(0, len(H), 1_000_000):               # in parts, for 20M
            h = H[s:s + 1_000_000]
            hi_[s:s + 1_000_000] = np.where(h >= 0, index(train[np.maximum(h, 0), 1], items), 0)
            hr_[s:s + 1_000_000] = np.where(h >= 0, r_tok[np.maximum(h, 0)], pad_r)
        t = lambda a: torch.tensor(a, device=DEV)            # noqa: E731
        return (t(index(rows[:, 0], users)), t(hi_), t(hr_), t(index(rows[:, 1], items)))

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if protocol == "temporal":
        # validation = newest 10% of each user
        by_time = np.lexsort((train[:, 3], train[:, 0]))
        su = train[by_time, 0]
        starts = np.r_[0, np.flatnonzero(np.diff(su)) + 1]
        counts = np.diff(np.r_[starts, len(su)])
        rank = np.arange(len(su)) - np.repeat(starts, counts)
        is_val = rank >= np.repeat((counts * 0.9).astype(int), counts)
        order = np.r_[rng.permutation(by_time[~is_val]), by_time[is_val]]
        cut = int((~is_val).sum())
    else:
        order = rng.permutation(len(train))
        cut = int(len(train) * 0.9)
    fit, val = torch.tensor(order[:cut], device=DEV), torch.tensor(order[cut:], device=DEV)
    TR = tensors(train, Htr)
    TE = tensors(test, Hte)
    Y = torch.tensor(train[:, 2], dtype=torch.float32, device=DEV)

    net = Net(kind, len(users), len(items), len(levels), float(train[order[:cut], 2].mean()))
    net = net.to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

    def predict(T, idx=None):
        net.eval()
        out = []
        n = len(T[0]) if idx is None else len(idx)
        with torch.no_grad():
            for s in range(0, n, 4096):
                b = slice(s, s + 4096) if idx is None else idx[s:s + 4096]
                out.append(net(*(x[b] for x in T)).cpu().numpy())
        return np.concatenate(out)

    batch = 2048 if len(train) > 1_000_000 else 256          # 20M: bigger batches
    t0 = time.time()
    best, best_state, wait, best_epoch = 9.0, None, 0, 0
    for epoch in range(30):
        net.train()
        perm = fit[torch.randperm(len(fit), device=DEV)]
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            loss = ((net(*(x[b] for x in TR)) - Y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = rmse(np.clip(predict(TR, val), lo, hi), train[val.cpu().numpy(), 2])
        if score < best - 1e-4:
            best, wait, best_epoch = score, 0, epoch + 1
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= 3:
                break
    net.load_state_dict(best_state)
    pred = np.clip(predict(TE), lo, hi)
    sec = time.time() - t0
    print(f"  {kind} test RMSE {rmse(pred, test[:, 2]):.4f}  MAE {mae(pred, test[:, 2]):.4f}  "
          f"val {best:.4f}  best epoch {best_epoch}  ({sec:.0f} s)", flush=True)
    with open(os.path.join(RESULTS, "rnn.jsonl"), "a") as f:
        f.write(json.dumps(dict(dataset=dataset, protocol=protocol, fold=fold,
                                model=kind, seed=seed, fingerprint=fp,
                                rmse=rmse(pred, test[:, 2]), mae=mae(pred, test[:, 2]),
                                val_rmse=best, best_epoch=best_epoch,
                                seconds=round(sec, 1), device=DEV)) + "\n")
    tag = protocol if fold is None else f"{protocol}{fold}"
    np.save(os.path.join(PREDS, f"pred_{dataset}_{tag}_{kind}_{seed}.npy"), pred)


if __name__ == "__main__":
    main()
