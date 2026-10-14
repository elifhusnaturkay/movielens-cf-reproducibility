"""The six co-clustering bases of Banerjee et al. (JMLR 2007), squared error.

g = cluster of user u, h = cluster of item i. a = global mean, a_u / a_i =
user / item mean, a_g / a_h = user / item cluster mean, a_gh = co-cluster mean,
a_uh = mean of user u over item cluster h, a_gi = mean of item i over user cluster g.

    C1  a_g + a_h - a
    C2  a_gh                                ||A - RR^T A CC^T||
    C3  a_gh + a_u - a_g
    C4  a_gh + a_i - a_h
    C5  a_gh + (a_u - a_g) + (a_i - a_h)    George & Merugu 2005
    C6  a_uh + a_gi - a_gh                  MSSRCC, Cho et al. 2004

Everything except the prediction rule is shared, so differences come from the
basis. Empty averages fall back to the next coarser one.
"""

import numpy as np

BASES = ["C1", "C2", "C3", "C4", "C5", "C6"]


class BasisCoClustering:
    def __init__(self, basis="C5", n_user_clusters=3, n_item_clusters=3, max_iter=50,
                 random_state=None, sequential=False):
        """sequential=True recomputes the averages between the user and the item step (Cho et al. 2004)."""
        assert basis in BASES
        self.basis, self.sequential = basis, sequential
        self.k, self.l = n_user_clusters, n_item_clusters
        self.max_iter, self.random_state = max_iter, random_state

    def fit(self, train, shape):
        """train: rows of (user, item, rating) with 0-based ids. shape: (n_users, n_items)."""
        self.u = train[:, 0].astype(np.int64)
        self.i = train[:, 1].astype(np.int64)
        self.r = train[:, 2].astype(np.float64)
        self.m, self.n = shape
        rng = np.random.default_rng(self.random_state)
        self.rho = rng.integers(self.k, size=self.m)      # user -> cluster
        self.gamma = rng.integers(self.l, size=self.n)    # item -> cluster

        self.mu = self.r.sum() / len(self.r)
        self.mu_u = self._mean(self.u, self.m, self.mu)
        self.mu_i = self._mean(self.i, self.n, self.mu)

        for it in range(self.max_iter):
            self._averages()
            if self.sequential:
                rho = self._assign_users()
                changed = (rho != self.rho).any()
                self.rho = rho
                self._averages()
                gamma = self._assign_items()
                changed = changed or (gamma != self.gamma).any()
                self.gamma = gamma
            else:
                rho, gamma = self._assign_users(), self._assign_items()
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
        return self._approx(u, i, self.rho[u], self.gamma[i])

    # the six prediction rules
    def _approx(self, u, i, g, h):
        b = self.basis
        if b == "C1":
            return self.A_g[g] + self.A_h[h] - self.mu
        if b == "C2":
            return self.A_gh[g, h]
        if b == "C3":
            return self.A_gh[g, h] + self.mu_u[u] - self.A_g[g]
        if b == "C4":
            return self.A_gh[g, h] + self.mu_i[i] - self.A_h[h]
        if b == "C5":
            return (self.A_gh[g, h] + self.mu_u[u] - self.A_g[g]
                    + self.mu_i[i] - self.A_h[h])
        # C6
        return self.A_uh[u, h] + self.A_gi[g, i] - self.A_gh[g, h]

    # averages over the observed ratings only
    def _mean(self, ids, size, fallback):
        count = np.bincount(ids, minlength=size)
        total = np.bincount(ids, weights=self.r, minlength=size)
        return np.where(count > 0, total / np.maximum(count, 1), fallback)

    def _averages(self):
        g, h = self.rho[self.u], self.gamma[self.i]
        k, l = self.k, self.l
        cell = np.bincount(g * l + h, minlength=k * l).reshape(k, l)
        tot = np.bincount(g * l + h, weights=self.r, minlength=k * l).reshape(k, l)
        div = lambda t, c, fb: np.where(c > 0, t / np.maximum(c, 1), fb)  # noqa: E731
        self.A_gh = div(tot, cell, self.mu)
        self.A_g = div(tot.sum(1), cell.sum(1), self.mu)
        self.A_h = div(tot.sum(0), cell.sum(0), self.mu)
        if self.basis == "C6":
            # user u over item cluster h: (m, l); item i over user cluster g: (k, n)
            uh = self.u * l + h
            c = np.bincount(uh, minlength=self.m * l).reshape(self.m, l)
            t = np.bincount(uh, weights=self.r, minlength=self.m * l).reshape(self.m, l)
            self.A_uh = div(t, c, self.mu_u[:, None])
            gi = g * self.n + self.i
            c = np.bincount(gi, minlength=k * self.n).reshape(k, self.n)
            t = np.bincount(gi, weights=self.r, minlength=k * self.n).reshape(k, self.n)
            self.A_gi = div(t, c, self.mu_i[None, :])

    # move each user / item to the cluster with the smallest error
    def _assign_users(self):
        h = self.gamma[self.i]
        errors = np.empty((self.k, self.m))
        for g in range(self.k):
            gg = np.full(len(self.r), g)
            e = self.r - self._approx(self.u, self.i, gg, h)
            errors[g] = np.bincount(self.u, weights=e ** 2, minlength=self.m)
        return errors.argmin(0)

    def _assign_items(self):
        g = self.rho[self.u]
        errors = np.empty((self.l, self.n))
        for h in range(self.l):
            hh = np.full(len(self.r), h)
            e = self.r - self._approx(self.u, self.i, g, hh)
            errors[h] = np.bincount(self.i, weights=e ** 2, minlength=self.n)
        return errors.argmin(0)
