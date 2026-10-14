"""Our own versions of two Surprise models, working on the list of ratings.

OwnSVD             surprise.SVD (Funk SVD, SGD) written from scratch. Same init,
                   same order of updates, same results (max diff 1e-15 on ml100k).
                   The SGD loop is compiled with numba.
SparseCoClustering George & Merugu (2005) co-clustering (basis C5 in bases.py)
                   with sums over observed ratings, so it also runs on 20M.
"""

import numpy as np
from numba import njit


# SVD
@njit(cache=True)
def _sgd(u, i, r, mu, bu, bi, pu, qi, n_epochs, lr, reg):
    n_factors = pu.shape[1]
    for _ in range(n_epochs):
        for k in range(len(r)):
            uu, ii = u[k], i[k]
            dot = 0.0
            for f in range(n_factors):
                dot += qi[ii, f] * pu[uu, f]
            err = r[k] - (mu + bu[uu] + bi[ii] + dot)
            bu[uu] += lr * (err - reg * bu[uu])
            bi[ii] += lr * (err - reg * bi[ii])
            for f in range(n_factors):
                puf = pu[uu, f]
                qif = qi[ii, f]
                pu[uu, f] += lr * (err * qif - reg * puf)
                qi[ii, f] += lr * (err * puf - reg * qif)


def _inner_ids(raw_ids):
    """Raw id -> inner id in order of first appearance (Surprise's rule)."""
    uniq, first, inv = np.unique(raw_ids, return_index=True, return_inverse=True)
    rank = np.empty(len(uniq), np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(uniq))
    return uniq, rank, rank[inv]


class OwnSVD:
    def __init__(self, n_factors=100, n_epochs=20, lr=0.005, reg=0.02,
                 init_std=0.1, rating_scale=(1, 5), random_state=42):
        self.n_factors, self.n_epochs = n_factors, n_epochs
        self.lr, self.reg, self.init_std = lr, reg, init_std
        self.scale, self.random_state = rating_scale, random_state

    def fit(self, train):
        self.users, self._urank, u = _inner_ids(train[:, 0])
        self.items, self._irank, i = _inner_ids(train[:, 1])
        order = np.argsort(u, kind="stable")        # user by user, file order inside
        u, i, r = u[order], i[order], train[order, 2].astype(np.float64)
        self.mu = np.mean(r)
        rng = np.random.RandomState(self.random_state)
        self.pu = rng.normal(0, self.init_std, (len(self.users), self.n_factors))
        self.qi = rng.normal(0, self.init_std, (len(self.items), self.n_factors))
        self.bu = np.zeros(len(self.users))
        self.bi = np.zeros(len(self.items))
        _sgd(u, i, r, self.mu, self.bu, self.bi, self.pu, self.qi,
             self.n_epochs, self.lr, self.reg)
        return self

    def _lookup(self, raw, known, rank):
        pos = np.clip(np.searchsorted(known, raw), 0, len(known) - 1)
        hit = known[pos] == raw
        return hit, np.where(hit, rank[pos], 0)

    def predict(self, test):
        ku, u = self._lookup(test[:, 0], self.users, self._urank)
        ki, i = self._lookup(test[:, 1], self.items, self._irank)
        est = self.mu + np.where(ku, self.bu[u], 0) + np.where(ki, self.bi[i], 0)
        dot = np.einsum("nf,nf->n", self.pu[u], self.qi[i])
        est += np.where(ku & ki, dot, 0)
        return np.clip(est, *self.scale)


# co-clustering
class SparseCoClustering:
    def __init__(self, n_user_clusters=3, n_item_clusters=3, max_iter=50,
                 random_state=None):
        self.k, self.l = n_user_clusters, n_item_clusters
        self.max_iter, self.random_state = max_iter, random_state

    def fit(self, train, shape):
        """train: rows of (user, item, rating) with 0-based ids. shape: (n_users, n_items)."""
        self.u = train[:, 0].astype(np.int64)
        self.i = train[:, 1].astype(np.int64)
        self.r = train[:, 2].astype(np.float64)
        m, n = shape
        rng = np.random.default_rng(self.random_state)
        self.rho = rng.integers(self.k, size=m)      # user -> cluster
        self.gamma = rng.integers(self.l, size=n)    # item -> cluster

        self.mu = self.r.sum() / len(self.r)
        self.mu_u = self._safe_mean(self.u, m)
        self.mu_i = self._safe_mean(self.i, n)

        for it in range(self.max_iter):
            self._averages()
            rho, gamma = self._assign_users(m), self._assign_items(n)
            changed = (rho != self.rho).any() or (gamma != self.gamma).any()
            self.rho, self.gamma = rho, gamma
            if not changed:
                break
        self.n_iter = it + 1
        self._averages()
        del self.u, self.i, self.r
        return self

    def predict(self, test):
        u, i = test[:, 0].astype(np.int64), test[:, 1].astype(np.int64)
        g, h = self.rho[u], self.gamma[i]
        return (self.A_cc[g, h] + self.mu_u[u] - self.A_uc[g]
                + self.mu_i[i] - self.A_ic[h])

    # sums over observed ratings only
    def _safe_mean(self, ids, size):
        count = np.bincount(ids, minlength=size)
        total = np.bincount(ids, weights=self.r, minlength=size)
        return np.where(count > 0, total / np.maximum(count, 1), self.mu)

    def _div(self, total, count):
        return np.where(count > 0, total / np.maximum(count, 1), self.mu)

    def _averages(self):
        cell = self.rho[self.u] * self.l + self.gamma[self.i]
        count = np.bincount(cell, minlength=self.k * self.l).reshape(self.k, self.l)
        total = np.bincount(cell, weights=self.r, minlength=self.k * self.l).reshape(self.k, self.l)
        self.A_cc = self._div(total, count)
        self.A_uc = self._div(total.sum(1), count.sum(1))
        self.A_ic = self._div(total.sum(0), count.sum(0))

    def _assign_users(self, m):
        d = self.r - self.mu_u[self.u]
        gi = self.gamma[self.i]
        errors = np.empty((self.k, m))
        for g in range(self.k):
            base = self.A_cc[g, gi] - self.A_uc[g] - self.A_ic[gi] + self.mu_i[self.i]
            errors[g] = np.bincount(self.u, weights=(d - base) ** 2, minlength=m)
        return errors.argmin(0)

    def _assign_items(self, n):
        d = self.r - self.mu_i[self.i]
        gu = self.rho[self.u]
        errors = np.empty((self.l, n))
        for h in range(self.l):
            base = self.A_cc[gu, h] - self.A_uc[gu] - self.A_ic[h] + self.mu_u[self.u]
            errors[h] = np.bincount(self.i, weights=(d - base) ** 2, minlength=n)
        return errors.argmin(0)
