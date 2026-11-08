"""Why does C6 (MSSRCC residue 2, Cho et al. 2004) get worse on MovieLens?

    python3 check_c6.py synthetic    # a few seconds
    python3 check_c6.py order        # a few seconds: parallel vs sequential update (Cho et al. 2004, Alg. 4.1)
    python3 check_c6.py shrink       # about 10 minutes, latest-small + ml-100k, 5-fold
    python3 check_c6.py shrink_big   # about 15 minutes, lam = 50, 100, 200
    python3 check_c6.py 20m          # 1-2 hours, C5 and C6 with the user side fixed
    python3 check_c6.py 20m_shrink   # about 10 minutes, C6 with shrinkage on 20M
    python3 check_c6.py table        # writes ../results/c6_check.md

1. synthetic: data that really has the C6 structure (a_uh + a_gi), 300 users x
   200 movies, 3 x 3 true groups, noise 0.3. We remove cells at random so that
   100, 50, 30, 20, 10, 5 and 2 % stay, and run all six bases (3 x 3, 80/20,
   3 seeds). If the code is right, C6 must win on full data. MovieLens is 1-6 % full.
2. order: the same C6 with the parallel update of our code and with the
   sequential update of Cho et al. 2004.
3. shrink: C6 with shrinkage. a_uh is pulled toward a_u - a_g + a_gh and a_gi
   toward a_i - a_h + a_gh, with weight lam against the number of ratings:
       a_uh' = (n_uh * a_uh + lam * prior) / (n_uh + lam)
   lam = 0 is C6, a very large lam gives C5. lam = 2, 5, 10, 25 (shrink) and
   50, 100, 200 (shrink_big) at 3 x 3 and 10 x 10, 5-fold. "val" picks lam on
   10 % of the train set, without the test set.
4. 20m: fix the user side, change the movie side. C5 and C6, K = 2, 3 and
   L = 1, 2, 3, 5, 7, 10, random 80/20. Rows go to ../results/grid.jsonl with
   note "C6 check". 20m_shrink: C6 with lam = 25 and 100 at 2 x 2 and 3 x 3.
"""

import json
import os
import sys
import time

import numpy as np

import run_bases as rb
from bases import BASES, BasisCoClustering

OUT = rb.OUT
DENSITIES = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
LAMS = [2, 5, 10, 25]
SIZES = [(3, 3), (10, 10)]
KS_20M, LS_20M = [2, 3], [1, 2, 3, 5, 7, 10]


class ShrunkC6(BasisCoClustering):
    def __init__(self, k, l, lam, **kw):
        super().__init__("C6", k, l, max_iter=50, random_state=42, **kw)
        self.lam = lam

    def _averages(self):
        super()._averages()
        g, h = self.rho[self.u], self.gamma[self.i]
        n_uh = np.bincount(self.u * self.l + h, minlength=self.m * self.l).reshape(self.m, self.l)
        prior = self.mu_u[:, None] - self.A_g[self.rho][:, None] + self.A_gh[self.rho]
        self.A_uh = (n_uh * self.A_uh + self.lam * prior) / (n_uh + self.lam)
        n_gi = np.bincount(g * self.n + self.i, minlength=self.k * self.n).reshape(self.k, self.n)
        prior = self.mu_i[None, :] - self.A_h[self.gamma][None, :] + self.A_gh[:, self.gamma]
        self.A_gi = (n_gi * self.A_gi + self.lam * prior) / (n_gi + self.lam)


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def write(name, row):
    with open(os.path.join(OUT, name), "a") as f:
        f.write(json.dumps(row) + "\n")


def synthetic():
    path = os.path.join(OUT, "c6_synthetic.jsonl")
    open(path, "w").close()
    m, n, k, l = 300, 200, 3, 3
    for seed in range(3):
        rng = np.random.default_rng(seed)
        g, h = rng.integers(k, size=m), rng.integers(l, size=n)
        a_uh, a_gi = rng.normal(3.5, 0.6, (m, l)), rng.normal(0, 0.6, (k, n))
        Z = a_uh[:, h] + a_gi[g, :] + rng.normal(0, 0.3, (m, n))
        uu, ii = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
        full = np.c_[uu.ravel(), ii.ravel(), Z.ravel()]
        for d in DENSITIES:
            data = full[rng.random(len(full)) < d]
            rng.shuffle(data)
            cut = int(0.8 * len(data))
            tr, te = data[:cut], data[cut:]
            row = dict(seed=seed, density=d, n_train=cut,
                       per_value_c6=round(cut / (m * l + k * n), 1))
            for b in BASES:
                mod = BasisCoClustering(b, k, l, random_state=42).fit(tr, (m, n))
                row[b] = rmse(mod.predict(te), te[:, 2])
            write("c6_synthetic.jsonl", row)
            print(row, flush=True)


def shrink():
    path = os.path.join(OUT, "c6_shrink.jsonl")
    have = rb.done(path, ["dataset", "fold", "k", "l"])
    for dataset in ["mlsmall", "ml100k"]:
        ds = rb.DATASETS[dataset]
        raw = ds["load"]()
        shape = (int(raw[:, 0].max()) + 1, int(raw[:, 1].max()) + 1)
        for fold in range(5):
            train, test = rb.get_split(raw, "kfold", fold)
            # validation part of the train set, for choosing lam without the test set
            rng = np.random.default_rng(fold)
            val = rng.random(len(train)) < 0.1
            for k, l in SIZES:
                if (dataset, fold, k, l) in have:
                    continue
                clip = lambda p: np.clip(p, *ds["scale"])  # noqa: E731
                row = dict(dataset=dataset, fold=fold, k=k, l=l, fingerprint=rb.cs.fingerprint(test))
                for b in ["C5", "C6"]:
                    mod = BasisCoClustering(b, k, l, random_state=42).fit(train, shape)
                    row[b] = rmse(clip(mod.predict(test)), test[:, 2])
                for lam in LAMS:
                    mod = ShrunkC6(k, l, lam).fit(train, shape)
                    row[f"lam{lam}"] = rmse(clip(mod.predict(test)), test[:, 2])
                vals = {lam: rmse(clip(ShrunkC6(k, l, lam).fit(train[~val], shape).predict(train[val])),
                                  train[val][:, 2]) for lam in LAMS}
                best = min(vals, key=vals.get)
                row["val_lam"] = best
                row["val"] = row[f"lam{best}"]
                write("c6_shrink.jsonl", row)
                print(row, flush=True)


def run_20m():
    path = os.path.join(OUT, "grid.jsonl")
    have = rb.done(path, ["dataset", "basis", "protocol", "k", "l"])
    ds = rb.DATASETS["ml20m"]
    raw = ds["load"]()
    shape = (int(raw[:, 0].max()) + 1, int(raw[:, 1].max()) + 1)
    train, test = rb.get_split(raw, "random", None)
    fp = rb.cs.fingerprint(test)
    print(f"[split] ml20m random fingerprint={fp}", flush=True)
    for k in KS_20M:
        for l in LS_20M:
            for b in ["C5", "C6"]:
                if ("ml20m", b, "random", k, l) in have:
                    continue
                row = dict(dataset="ml20m", protocol="random", fold=None, fingerprint=fp,
                           **rb.run_one(b, k, l, train, test, shape, ds["scale"]), note="C6 check")
                print(f"  {b} {k} x {l:<2} RMSE={row['rmse']:.4f}  train={row['train_rmse']:.4f}  {row['fit_s']}s",
                      flush=True)
                with open(path, "a") as f:
                    f.write(json.dumps(row) + "\n")


def order():
    """latest small, fold 0: C5 and C6 with the parallel update and with the
    sequential update of Cho et al. 2004 (Algorithm 4.1)."""
    raw = rb.DATASETS["mlsmall"]["load"]()
    shape = (int(raw[:, 0].max()) + 1, int(raw[:, 1].max()) + 1)
    tr, te = rb.get_split(raw, "kfold", 0)
    for k, l in SIZES:
        for seq in [False, True]:
            for b in ["C5", "C6"]:
                mod = BasisCoClustering(b, k, l, random_state=42, sequential=seq).fit(tr, shape)
                row = dict(k=k, l=l, sequential=seq, basis=b, train=rmse(mod.predict(tr), tr[:, 2]),
                           test=rmse(np.clip(mod.predict(te), 0.5, 5), te[:, 2]), n_iter=mod.n_iter)
                write("c6_order.jsonl", row)
                print(row, flush=True)


LAMS_BIG = [50, 100, 200]


def shrink_big():
    """In shrink(), validation always chose the largest lam (25). Same folds,
    larger lam; validation now chooses among 25, 50, 100, 200."""
    path = os.path.join(OUT, "c6_shrink_big.jsonl")
    have = rb.done(path, ["dataset", "fold", "k", "l"])
    for dataset in ["mlsmall", "ml100k"]:
        ds = rb.DATASETS[dataset]
        raw = ds["load"]()
        shape = (int(raw[:, 0].max()) + 1, int(raw[:, 1].max()) + 1)
        for fold in range(5):
            train, test = rb.get_split(raw, "kfold", fold)
            val = np.random.default_rng(fold).random(len(train)) < 0.1
            for k, l in SIZES:
                if (dataset, fold, k, l) in have:
                    continue
                clip = lambda p: np.clip(p, *ds["scale"])  # noqa: E731
                row = dict(dataset=dataset, fold=fold, k=k, l=l, fingerprint=rb.cs.fingerprint(test))
                for lam in LAMS_BIG:
                    row[f"lam{lam}"] = rmse(clip(ShrunkC6(k, l, lam).fit(train, shape).predict(test)), test[:, 2])
                vals = {lam: rmse(clip(ShrunkC6(k, l, lam).fit(train[~val], shape).predict(train[val])),
                                  train[val][:, 2]) for lam in [25] + LAMS_BIG}
                row["val_lam"] = min(vals, key=vals.get)
                write("c6_shrink_big.jsonl", row)
                print(row, flush=True)


def shrink_20m():
    """C6 with shrinkage on 20M (random 80/20), lam = 25 and 100, at 2 x 2 and 3 x 3."""
    path = os.path.join(OUT, "c6_shrink_20m.jsonl")
    have = rb.done(path, ["k", "l", "lam"])
    ds = rb.DATASETS["ml20m"]
    raw = ds["load"]()
    shape = (int(raw[:, 0].max()) + 1, int(raw[:, 1].max()) + 1)
    train, test = rb.get_split(raw, "random", None)
    fp = rb.cs.fingerprint(test)
    for k, l in [(2, 2), (3, 3)]:
        for lam in [25, 100]:
            if (k, l, lam) in have:
                continue
            t = time.time()
            mod = ShrunkC6(k, l, lam).fit(train, shape)
            row = dict(k=k, l=l, lam=lam, fingerprint=fp, rmse=rmse(mod.predict(test), test[:, 2]),
                       rmse_clip=rmse(np.clip(mod.predict(test), *ds["scale"]), test[:, 2]),
                       fit_s=round(time.time() - t, 1))
            write("c6_shrink_20m.jsonl", row)
            print(row, flush=True)


def table():
    lines = ["# C6 check (7 Nov)", ""]
    p = os.path.join(OUT, "c6_synthetic.jsonl")
    if os.path.exists(p):
        rows = [json.loads(x) for x in open(p)]
        lines += ["## 1. Synthetic data with the C6 structure (mean of 3 seeds, test RMSE, noise = 0.3)", "",
                  "| Density | Train ratings per C6 value | " + " | ".join(BASES) + " |",
                  "|---|---|" + "---|" * len(BASES)]
        for d in DENSITIES:
            r = [x for x in rows if x["density"] == d]
            means = {b: np.mean([x[b] for x in r]) for b in BASES}
            best = min(means, key=means.get)
            cells = [f"**{means[b]:.3f}**" if b == best else f"{means[b]:.3f}" for b in BASES]
            lines.append(f"| {d:.0%} | {np.mean([x['per_value_c6'] for x in r]):.1f} | " + " | ".join(cells) + " |")
        lines.append("")
    p = os.path.join(OUT, "c6_shrink.jsonl")
    if os.path.exists(p):
        rows = [json.loads(x) for x in open(p)]
        cols = ["C5", "C6"] + [f"lam{x}" for x in LAMS] + ["val"]
        lines += ["## 2. C6 with shrinkage (5-fold mean test RMSE)", "",
                  "| Dataset | Size | " + " | ".join(c.replace("lam", "λ = ") for c in cols) + " |",
                  "|---|---|" + "---|" * len(cols)]
        for ds in ["mlsmall", "ml100k"]:
            for k, l in SIZES:
                r = [x for x in rows if x["dataset"] == ds and x["k"] == k and x["l"] == l]
                if len(r) < 5:
                    continue
                lines.append(f"| {ds} | {k} × {l} | " + " | ".join(f"{np.mean([x[c] for x in r]):.4f}" for c in cols) + " |")
        lines.append("")
    rows = [json.loads(x) for x in open(os.path.join(OUT, "grid.jsonl"))]
    m = {(r["basis"], r["k"], r["l"]): r for r in rows if r["dataset"] == "ml20m" and r["protocol"] == "random"}
    lines += ["## 3. MovieLens 20M, user side fixed (random 80/20, test RMSE, train RMSE in brackets)", "",
              "| K × L | C5 | C6 |", "|---|---|---|"]
    for k in KS_20M:
        for l in LS_20M:
            cells = []
            for b in ["C5", "C6"]:
                r = m.get((b, k, l))
                tr = f" ({r['train_rmse']:.4f})" if r and r.get("train_rmse") is not None else ""
                cells.append(f"{r['rmse']:.4f}{tr}" if r else "–")
            lines.append(f"| {k} × {l} | " + " | ".join(cells) + " |")
    with open(os.path.join(OUT, "c6_check.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    {"synthetic": synthetic, "shrink": shrink, "20m": run_20m, "order": order, "shrink_big": shrink_big, "20m_shrink": shrink_20m, "table": table}[sys.argv[1]]()
