"""Features of one (user, movie) pair, for SVR and boosting.

The ratings matrix has no features, so we make them from the train ratings:

  0  user mean      (shrunk to the global mean, so few ratings -> near mu)
  1  user count     (log)
  2  user std
  3  movie mean     (shrunk)
  4  movie count    (log)
  5  movie std
  6  genre taste    how much this user likes this movie's genres:
                    mean of (rating - user mean) over the user's movies of the
                    same genre, averaged over the movie's genres
  7+ genres         one 0/1 column per genre

Every number is computed from the rows given to fit() only. If the rating we
predict were inside its own movie mean, the model would see the answer
(leakage). A user or movie not in fit() gets the global values.
"""

import numpy as np

SHRINK = 5     # a mean of n ratings counts like n / (n + 5)


def _group_stats(ids, values, size):
    count = np.bincount(ids, minlength=size).astype(float)
    total = np.bincount(ids, values, minlength=size)
    square = np.bincount(ids, values ** 2, minlength=size)
    mean = np.divide(total, count, out=np.zeros(size), where=count > 0)
    var = np.divide(square, count, out=np.zeros(size), where=count > 0) - mean ** 2
    return count, total, np.sqrt(np.maximum(var, 0))


class Features:
    def __init__(self, genres):
        self.G = genres                      # (n_movies, n_genres), row = movie id

    def fit(self, train, shape):
        u = train[:, 0].astype(np.int64)
        i = train[:, 1].astype(np.int64)
        r = train[:, 2].astype(float)
        m, n = shape
        self.mu = r.mean()

        cu, su, self.u_std = _group_stats(u, r, m)
        ci, si, self.i_std = _group_stats(i, r, n)
        self.u_mean = (su + SHRINK * self.mu) / (cu + SHRINK)
        self.i_mean = (si + SHRINK * self.mu) / (ci + SHRINK)
        self.u_cnt, self.i_cnt = np.log1p(cu), np.log1p(ci)

        # genre taste: (user, genre) mean of the user's residual
        res = r - self.u_mean[u]
        g = self.G[i]                                   # (rows, n_genres)
        k = self.G.shape[1]
        top = np.zeros((m, k))
        bot = np.zeros((m, k))
        for c in range(k):                              # bincount: fast on 20M
            top[:, c] = np.bincount(u, g[:, c] * res, minlength=m)
            bot[:, c] = np.bincount(u, g[:, c], minlength=m)
        self.taste = top / (bot + SHRINK)               # shrunk to 0
        return self

    def transform(self, rows):
        u = rows[:, 0].astype(np.int64)
        i = rows[:, 1].astype(np.int64)
        g = self.G[i]
        n_g = np.maximum(g.sum(1), 1)
        taste = (self.taste[u] * g).sum(1) / n_g
        X = np.column_stack([self.u_mean[u], self.u_cnt[u], self.u_std[u],
                             self.i_mean[i], self.i_cnt[i], self.i_std[i],
                             taste, g])
        return X.astype(np.float32)
