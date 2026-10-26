"""Item-based KNN with baselines, the same as surprise.KNNBaseline.

    KNNBaseline(k=40, sim_options={"name": "pearson_baseline", "user_based": False})

Why our own code: Surprise keeps 5 full item x item matrices in float64.
On MovieLens 20M (about 26,000 movies) that does not fit in 16 GB memory.
Here every row of the similarity matrix is computed alone (numba, in
parallel), so only the similarity matrix itself is kept. With float32 it
is 2.7 GB on 20M.

The steps are the same as in Surprise (similarities.pyx, knns.py,
optimize_baselines.pyx):
  1. baselines bu, bi with 10 rounds of ALS (reg_u = 15, reg_i = 10)
  2. d = r - (mu + bu + bi) for every train rating
  3. sim(i, j) over the common users U_ij (at least 2):
       sum d_ui d_uj / sqrt(sum d_ui^2 * sum d_uj^2) * (|U_ij| - 1) / (|U_ij| - 1 + 100)
  4. prediction: mu + bu + bi, plus the weighted mean of d over the 40 most
     similar movies the user rated (only neighbors with sim > 0 count)
  5. unknown user or movie: only the baseline. Clipped to the rating scale.
"""

import numpy as np
from numba import njit, prange

K = 40
SHRINK = 100.0


def _ids(raw, known):
    pos = np.clip(np.searchsorted(known, raw), 0, len(known) - 1)
    hit = known[pos] == raw
    return hit, np.where(hit, pos, 0)


def _csr(rows, cols, vals, n_rows):
    """Group (row, col, val) by row: start index per row, cols and vals."""
    order = np.argsort(rows, kind="stable")
    start = np.zeros(n_rows + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=n_rows), out=start[1:])
    return start, cols[order].astype(np.int64), vals[order]


@njit(parallel=True, cache=True)
def _similarity(n_items, i_start, i_users, u_start, u_items, u_dev, i_dev_by_user, out):
    """One row of the similarity matrix per item, rows in parallel.

    i_start/i_users: users of each item. u_start/u_items/u_dev: items and
    residuals of each user. i_dev_by_user: residual of (item, user) in the
    same order as i_users."""
    for a in prange(n_items):
        freq = np.zeros(n_items)
        prods = np.zeros(n_items)
        sq_a = np.zeros(n_items)
        sq_b = np.zeros(n_items)
        for p in range(i_start[a], i_start[a + 1]):
            u = i_users[p]
            da = i_dev_by_user[p]
            for q in range(u_start[u], u_start[u + 1]):
                b = u_items[q]
                db = u_dev[q]
                freq[b] += 1
                prods[b] += da * db
                sq_a[b] += da * da
                sq_b[b] += db * db
        for b in range(n_items):
            if b == a:
                out[a, b] = 1.0
            elif freq[b] < 2:
                out[a, b] = 0.0
            else:
                den = np.sqrt(sq_a[b] * sq_b[b])
                s = prods[b] / den if den > 0 else 0.0
                out[a, b] = s * (freq[b] - 1) / (freq[b] - 1 + SHRINK)


@njit(parallel=True, cache=True)
def _predict(users, items, ok, base, sim, u_start, u_items, u_dev, k, out):
    for n in prange(len(users)):
        out[n] = base[n]
        if not ok[n]:
            continue
        u, i = users[n], items[n]
        a, b = u_start[u], u_start[u + 1]
        s = np.empty(b - a)
        for q in range(a, b):
            s[q - a] = sim[i, u_items[q]]
        order = np.argsort(-s, kind="mergesort")    # stable, like heapq.nlargest
        top, sw = 0.0, 0.0
        for t in range(min(k, b - a)):
            q = order[t]
            if s[q] > 0:
                sw += s[q]
                top += s[q] * u_dev[a + q]
        if sw > 0:
            out[n] += top / sw


class OwnKNN:
    def __init__(self, k=K, rating_scale=(1, 5), dtype=np.float64):
        self.k, self.scale, self.dtype = k, rating_scale, dtype

    def fit(self, train):
        self.users, u = np.unique(train[:, 0], return_inverse=True)
        self.items, i = np.unique(train[:, 1], return_inverse=True)
        r = train[:, 2].astype(np.float64)
        m, n = len(self.users), len(self.items)
        self.mu = r.mean()

        # 1. ALS baselines, as baseline_als() in Surprise
        cu, ci = np.bincount(u, minlength=m), np.bincount(i, minlength=n)
        self.bu, self.bi = np.zeros(m), np.zeros(n)
        for _ in range(10):
            self.bi = np.bincount(i, r - self.mu - self.bu[u], n) / (10 + ci)
            self.bu = np.bincount(u, r - self.mu - self.bi[i], m) / (15 + cu)

        # 2. residuals, grouped by user and by item
        d = r - (self.mu + self.bu[u] + self.bi[i])
        self.u_start, self.u_items, self.u_dev = _csr(u, i, d, m)
        i_start, i_users, i_dev = _csr(i, u, d, n)

        # 3. similarity matrix
        self.sim = np.empty((n, n), self.dtype)
        _similarity(n, i_start, i_users, self.u_start, self.u_items, self.u_dev,
                    i_dev, self.sim)
        return self

    def predict(self, rows):
        ku, u = _ids(rows[:, 0], self.users)
        ki, i = _ids(rows[:, 1], self.items)
        base = self.mu + np.where(ku, self.bu[u], 0) + np.where(ki, self.bi[i], 0)
        out = np.empty(len(rows))
        _predict(u, i, ku & ki, base, self.sim, self.u_start, self.u_items, self.u_dev,
                 self.k, out)
        return np.clip(out, *self.scale)
