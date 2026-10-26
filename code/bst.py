"""Two small transformers for rating prediction.

    python3 bst.py mlsmall temporal bst 42
    python3 bst.py ml100k kfold 0 bert 42

Two models, the same data pipeline:
  bst   our own small transformer, plain PyTorch (BST style, Chen et al. 2019)
  bert  the same, but the encoder is a Hugging Face BertModel (random start)

Input of one rating (user u, movie i, time t): the last L=20 ratings of u in
the TRAIN set that come before t. Each token = movie embedding + rating
embedding + position. The last token is movie i with a [MASK] rating. The
encoder output of this token, together with a user embedding, goes through
an MLP to one number. Prediction = global mean + user bias + movie bias + MLP.

No leakage: histories are built from the train set only, and a test rating
only sees train ratings with an earlier time (checked with assert).
Early stopping on 10% of train. The test set is used once, at the end.
Results go to ../results/transformer.jsonl, test predictions to
../results/preds/.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
import common_split as cs  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
PREDS = os.path.join(RESULTS, "preds")
os.makedirs(PREDS, exist_ok=True)
DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")
L = 20          # history length
D = 64          # embedding size
DATASETS = {
    "mlsmall": dict(load=cs.load_mlsmall, scale=(0.5, 5)),
    "ml100k": dict(load=cs.load_ml100k_timed, scale=(1, 5)),
    "ml20m": dict(load=cs.load_ml20m, scale=(0.5, 5)),
}


def get_split(raw, protocol, fold):
    if protocol == "random":
        return cs.split(raw)
    if protocol == "temporal":
        return cs.temporal_split(raw)
    if protocol == "kfold":
        return list(cs.kfold(raw, 5))[fold]
    sys.exit(f"unknown split {protocol}")


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


# ---- histories -------------------------------------------------------------

def index(ids, known):
    """raw id -> 1..n if seen in train, 0 if not."""
    pos = np.clip(np.searchsorted(known, ids), 0, len(known) - 1)
    return np.where(known[pos] == ids, pos + 1, 0)


def histories(train, rows, is_train):
    """For every row: positions (into the time-sorted train set) of the last L
    train ratings of the same user before this row. -1 = empty slot.

    is_train=True: rows are the train set itself; a rating sees the ratings
    before it in the sorted order (so never itself).
    is_train=False: a rating sees the train ratings with a strictly earlier time.
    """
    order = np.lexsort((train[:, 3], train[:, 0]))          # by user, then time
    su, st = train[order, 0], train[order, 3]
    users, starts = np.unique(su, return_index=True)
    ends = np.r_[starts[1:], len(su)]
    H = np.full((len(rows), L), -1, np.int32)               # int32: 20M needs less memory
    if is_train:
        rank = np.empty(len(order), np.int64)
        rank[order] = np.arange(len(order))
        pos_of_row = rank                                    # row k sits at rank[k]
    for k, (u, t) in enumerate(zip(rows[:, 0], rows[:, 3])):
        j = np.searchsorted(users, u)
        if j == len(users) or users[j] != u:
            continue                                         # user not in train
        a, b = starts[j], ends[j]
        if is_train:
            stop = pos_of_row[k]
        else:
            stop = a + np.searchsorted(st[a:b], t, side="left")
        h = np.arange(max(a, stop - L), stop)
        H[k, L - len(h):] = order[h]                         # right aligned
    return H


# ---- the model ---------------------------------------------------------------

class Net(nn.Module):
    def __init__(self, kind, n_users, n_items, n_levels, mu):
        super().__init__()
        self.kind, self.mu = kind, mu
        self.item = nn.Embedding(n_items + 1, D, padding_idx=0)
        self.rate = nn.Embedding(n_levels + 2, D)            # levels, [MASK], [PAD]
        self.user = nn.Embedding(n_users + 1, D, padding_idx=0)
        self.ub = nn.Embedding(n_users + 1, 1, padding_idx=0)
        self.ib = nn.Embedding(n_items + 1, 1, padding_idx=0)
        nn.init.zeros_(self.ub.weight)
        nn.init.zeros_(self.ib.weight)
        if kind == "bst":
            self.pos = nn.Embedding(L + 1, D)
            layer = nn.TransformerEncoderLayer(D, 4, 4 * D, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        else:
            from transformers import BertConfig, BertModel
            cfg = BertConfig(vocab_size=1, hidden_size=D, num_hidden_layers=2,
                             num_attention_heads=4, intermediate_size=4 * D,
                             max_position_embeddings=L + 1, hidden_dropout_prob=0.1,
                             attention_probs_dropout_prob=0.1)
            self.enc = BertModel(cfg)                        # has its own positions
        self.head = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(64, 1))

    def forward(self, u, hist_i, hist_r, target):
        # tokens: L history tokens + 1 target token
        items = torch.cat([hist_i, target[:, None]], 1)
        mask_r = torch.full_like(target, self.rate.num_embeddings - 2)
        rates = torch.cat([hist_r, mask_r[:, None]], 1)
        x = self.item(items) + self.rate(rates)
        pad = torch.cat([hist_i == 0, torch.zeros_like(target, dtype=torch.bool)[:, None]], 1)
        if self.kind == "bst":
            x = x + self.pos.weight[None]
            h = self.enc(x, src_key_padding_mask=pad)[:, -1]
        else:
            h = self.enc(inputs_embeds=x, attention_mask=(~pad).long()).last_hidden_state[:, -1]
        out = self.head(torch.cat([h, self.user(u)], 1)).squeeze(1)
        return self.mu + self.ub(u).squeeze(1) + self.ib(target).squeeze(1) + out


def main():
    dataset, protocol = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    fold = int(rest.pop(0)) if protocol == "kfold" else None
    kind, seed = rest[0], int(rest[1])
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
    # leakage check: a test rating only sees earlier train ratings of the same user
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
        for s in range(0, len(H), 1_000_000):               # in parts: less memory on 20M
            h = H[s:s + 1_000_000]
            hi_[s:s + 1_000_000] = np.where(h >= 0, index(train[np.maximum(h, 0), 1], items), 0)
            hr_[s:s + 1_000_000] = np.where(h >= 0, r_tok[np.maximum(h, 0)], pad_r)
        t = lambda a: torch.tensor(a, device=DEV)            # noqa: E731
        return (t(index(rows[:, 0], users)), t(hi_), t(hr_), t(index(rows[:, 1], items)))

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if protocol == "temporal":
        # the test set is the future, so the validation set is too:
        # the newest 10% of each user's train ratings
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
    with open(os.path.join(RESULTS, "transformer.jsonl"), "a") as f:
        f.write(json.dumps(dict(dataset=dataset, protocol=protocol, fold=fold,
                                model=kind, seed=seed, fingerprint=fp,
                                rmse=rmse(pred, test[:, 2]), mae=mae(pred, test[:, 2]),
                                val_rmse=best, best_epoch=best_epoch,
                                seconds=round(sec, 1))) + "\n")
    tag = protocol if fold is None else f"{protocol}{fold}"
    np.save(os.path.join(PREDS, f"pred_{dataset}_{tag}_{kind}_{seed}.npy"), pred)


if __name__ == "__main__":
    main()
