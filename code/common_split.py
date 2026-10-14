"""Datasets, split protocols and test-set fingerprints, shared by all scripts.

Data is read from ~/.surprise_data (or $CFREPRO_DATA), see scripts/download_data.py.
"""

import hashlib
import os

import numpy as np

DATA_DIR = os.path.expanduser(os.environ.get("CFREPRO_DATA", "~/.surprise_data"))
PATH = os.path.join(DATA_DIR, "ml-100k/ml-100k/u.data")
# MovieLens 20M, the paper's Dataset 2. Either the Kaggle copy
# (grouplens/movielens-20m-dataset: rating.csv, text timestamps) in ml-20m-kaggle/,
# or the GroupLens zip (ratings.csv, unix timestamps) in ml-20m/.
ML20M_DIR = os.path.join(DATA_DIR, "ml-20m-kaggle")
ML20M_GROUPLENS_DIR = os.path.join(DATA_DIR, "ml-20m")
# MovieLens latest small from files.grouplens.org (ml-latest-small.zip).
# The paper's Dataset 1: 610 users x 9,742 movies.
MLSMALL_DIR = os.path.join(DATA_DIR, "ml-latest-small")

SEED = 42
TEST_SIZE = 0.2


def load():
    """MovieLens-100k as an (n, 3) int array of (user, item, rating)."""
    raw = np.loadtxt(PATH, dtype=int)[:, :3]
    raw[:, :2] -= 1  # ids are 1-based
    return raw


def load_ml100k_timed():
    """Same as load(), with the timestamp as a 4th column."""
    raw = np.loadtxt(PATH, dtype=np.int64)
    raw[:, :2] -= 1  # ids are 1-based
    return raw.astype(float)


def load_mlsmall():
    """MovieLens latest-small: (user, item, rating, time), 100,836 rows."""
    path = os.path.join(MLSMALL_DIR, "ratings.csv")
    return np.loadtxt(path, delimiter=",", skiprows=1)


def load_ml20m():
    """MovieLens 20M: (user, item, rating, time). Cached as .npy after the first read."""
    import pandas as pd
    kaggle = os.path.join(ML20M_DIR, "rating.csv")
    folder = ML20M_DIR if os.path.exists(kaggle) else ML20M_GROUPLENS_DIR
    cache = os.path.join(folder, "rating.npy")
    if os.path.exists(cache):
        return np.load(cache)
    if os.path.exists(kaggle):
        df = pd.read_csv(kaggle)
        t = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
        t = t.astype("int64").to_numpy() // 10**9
    else:
        df = pd.read_csv(os.path.join(ML20M_GROUPLENS_DIR, "ratings.csv"))
        t = df["timestamp"].to_numpy()
    raw = np.column_stack([df["userId"].to_numpy(float), df["movieId"].to_numpy(float),
                           df["rating"].to_numpy(float), t])
    np.save(cache, raw)
    return raw


# protocols
# each protocol returns (train, test) with the same columns as raw


def split(raw, test_size=TEST_SIZE, seed=SEED):
    """Random 80/20: one NumPy permutation with the seed, first 80% train."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(raw))
    cut = int(len(raw) * (1 - test_size))
    return raw[order[:cut]], raw[order[cut:]]


def temporal_split(raw, test_size=TEST_SIZE):
    """Per user: oldest 80% train, newest 20% test (paper, p. 6)."""
    order = np.lexsort((raw[:, 3], raw[:, 0]))          # by user, then by time
    users = raw[order, 0]
    starts = np.r_[0, np.flatnonzero(np.diff(users)) + 1]
    counts = np.diff(np.r_[starts, len(users)])
    rank = np.arange(len(users)) - np.repeat(starts, counts)
    cut = np.repeat((counts * (1 - test_size)).astype(int), counts)
    is_test = rank >= cut
    return raw[order[~is_test]], raw[order[is_test]]


def leave_one_out(raw, last=True, seed=SEED):
    """One test rating per user: the newest one (last=True) or a random one."""
    if last:
        order = np.lexsort((raw[:, 3], raw[:, 0]))          # by user, then by time
    else:
        r = np.random.default_rng(seed).random(len(raw))
        order = np.lexsort((r, raw[:, 0]))                 # by user, then at random
    users = raw[order, 0]
    ends = np.r_[np.flatnonzero(np.diff(users)), len(users) - 1]
    is_test = np.zeros(len(raw), bool)
    is_test[order[ends]] = True
    return raw[~is_test], raw[is_test]


def kfold(raw, k=5, seed=SEED):
    """k-fold CV: yields (train, test) k times from one permutation."""
    order = np.random.default_rng(seed).permutation(len(raw))
    for part in np.array_split(order, k):
        mask = np.zeros(len(raw), bool)
        mask[part] = True
        yield raw[~mask], raw[mask]


def load_split(test_size=TEST_SIZE, seed=SEED):
    """load() + split() in one call, the usual entry point."""
    return split(load(), test_size, seed)


# test-set fingerprint
def fingerprint(test):
    """First 12 hex digits of sha256 over the sorted test ratings."""
    rows = sorted((int(t[0]), int(t[1]), float(t[2])) for t in test)
    text = ";".join(f"{u},{i},{r:g}" for u, i, r in rows)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def print_header(script, train, test):
    """The line every script in this folder prints before it does any work."""
    print(f"[split] {script}  random {int((1 - TEST_SIZE) * 100)}/"
          f"{int(TEST_SIZE * 100)}  seed={SEED}  train={len(train)}  "
          f"test={len(test)}  fingerprint={fingerprint(test)}")


# Surprise format
def to_surprise(train, test, rating_scale=(1, 5)):
    """Surprise trainset and testset from our arrays. Use rating_scale=(0.5, 5) for half stars."""
    import pandas as pd
    from surprise import Dataset, Reader

    reader = Reader(rating_scale=rating_scale)
    frame = pd.DataFrame(train[:, :3]) if train.shape[1] > 3 else pd.DataFrame(train)
    trainset = Dataset.load_from_df(frame, reader).build_full_trainset()
    testset = [(t[0], t[1], t[2]) for t in test]
    return trainset, testset
