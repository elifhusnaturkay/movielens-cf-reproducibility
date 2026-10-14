"""Tables and one plot for svd_full.py.

    python3 plot_svd_full.py

Markdown tables on screen; results/svd_hparams.png (Funk SVD, one setting at a
time, test RMSE filled and train RMSE hollow, 5-fold mean).
"""

import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TITLES = {"mlsmall": "Dataset 1 (ml-latest-small)", "ml100k": "MovieLens 100k",
          "ml20m": "Dataset 2 (MovieLens 20M)"}
PROTO = {"random": "random 80/20", "temporal": "temporal 80/20", "kfold": "5-fold CV",
         "loo_last": "leave-last-one-out", "loo_random": "leave-random-one-out"}
DEFAULT = {"reg": 0.02, "n_epochs": 20, "lr": 0.005}


def load(name):
    path = os.path.join(OUT, name)
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def protocol_tables(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["model"], r["k"], r["protocol"])].append(r["rmse"])
    for ds in ("mlsmall", "ml100k", "ml20m"):
        cols = [(m, k) for m in ("tsvd", "funk") for k in (5, 15, 50, 100)
                if any(key[:3] == (ds, m, k) for key in by)]
        if not cols:
            continue
        print(f"\n### {TITLES[ds]}\n")
        print("| split | " + " | ".join(f"{'trunc.' if m == 'tsvd' else 'Funk'} K={k}"
                                       for m, k in cols) + " |")
        print("|---" * (len(cols) + 1) + "|")
        for p in PROTO:
            cells = []
            for m, k in cols:
                v = by.get((ds, m, k, p))
                cells.append("" if not v else f"{np.mean(v):.4f}")
            print(f"| {PROTO[p]} | " + " | ".join(cells) + " |")


def hparam_plot(rows):
    dss = [d for d in ("mlsmall", "ml100k") if any(r["dataset"] == d for r in rows)]
    if not dss:
        return
    fig, axes = plt.subplots(len(dss), 3, figsize=(10, 3.1 * len(dss)), dpi=150, squeeze=False)
    for a, ds in enumerate(dss):
        for b, param in enumerate(("reg", "n_epochs", "lr")):
            ax = axes[a, b]
            sub = [r for r in rows if r["dataset"] == ds and r["param"] == param]
            vals = sorted({r["value"] for r in sub})
            te = [np.mean([r["rmse"] for r in sub if r["value"] == v]) for v in vals]
            tr = [np.mean([r["train_rmse"] for r in sub if r["value"] == v]) for v in vals]
            x = np.arange(len(vals))
            ax.plot(x, te, color=BLUE, marker="o", linewidth=2, label="test")
            ax.plot(x, tr, color=ORANGE, marker="o", markerfacecolor="none", linewidth=1.5,
                    linestyle="--", label="train")
            d = vals.index(DEFAULT[param])
            ax.axvline(d, color=MUTED, linestyle=":", linewidth=1)
            best = int(np.argmin(te))
            ax.annotate(f"{te[best]:.4f}", (x[best], te[best]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color=INK)
            ax.set_xticks(x, [f"{v:g}" for v in vals], fontsize=7)
            ax.set_title(f"{TITLES[ds]}: {param} (dotted = default)", fontsize=8,
                         color=INK, loc="left")
            ax.grid(axis="y", color=GRID)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=7)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Funk SVD (K = 100), one setting changed at a time, 5-fold mean RMSE",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    path = os.path.join(OUT, "figures", "svd_hparams.png")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def hparam_tables(rows):
    for ds in ("mlsmall", "ml100k", "ml20m"):
        sub = [r for r in rows if r["dataset"] == ds]
        if not sub:
            continue
        print(f"\n### {TITLES[ds]}, Funk SVD settings\n")
        print("| setting | values → test RMSE (train RMSE) |")
        print("|---|---|")
        for param in ("reg", "n_epochs", "lr"):
            ps = [r for r in sub if r["param"] == param]
            if not ps:
                continue
            vals = sorted({r["value"] for r in ps})
            cells = []
            for v in vals:
                te = np.mean([r["rmse"] for r in ps if r["value"] == v])
                tr = np.mean([r["train_rmse"] for r in ps if r["value"] == v])
                mark = "**" if v == DEFAULT[param] else ""
                cells.append(f"{mark}{v:g}{mark}: {te:.4f} ({tr:.3f})")
            print(f"| {param} | " + " · ".join(cells) + " |")


def main():
    protocol_tables(load("svd_protocols.jsonl"))
    hp = load("svd_hparams.jsonl")
    hparam_plot(hp)
    hparam_tables(hp)


if __name__ == "__main__":
    main()
