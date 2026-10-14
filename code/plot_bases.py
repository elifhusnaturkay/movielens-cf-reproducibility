"""Plots and tables for the six bases, from results/protocols.jsonl and grid.jsonl.

    python3 plot_bases.py           # PNGs into results/, markdown tables on screen

bases_3x3.png        3 x 3, every basis, test RMSE (filled) and train RMSE (hollow)
bases_square_<ds>.png  K = L from 1 to 50, one line per basis
bases_heat_<ds>.png    the full K x L grid, one small heat map per basis
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
BASES = ["C1", "C2", "C3", "C4", "C5", "C6"]
COLOR = dict(zip(BASES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MARK = dict(zip(BASES, ["o", "s", "^", "v", "D", "P"]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"C1": "C1", "C2": "C2 (co-cluster mean)", "C3": "C3", "C4": "C4",
         "C5": "C5 (George & Merugu 2005)", "C6": "C6 (MSSRCC, Cho et al. 2004)"}
TITLES = {"mlsmall": "Dataset 1 (ml-latest-small)", "ml100k": "MovieLens 100k",
          "ml20m": "Dataset 2 (MovieLens 20M)"}
PROTO = {"random": "random 80/20", "temporal": "temporal 80/20", "kfold": "5-fold CV",
         "loo_last": "leave-last-one-out", "loo_random": "leave-random-one-out"}


def load(name):
    path = os.path.join(OUT, name)
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def mean_by(rows, keys, val="rmse"):
    by = collections.defaultdict(list)
    for r in rows:
        if r.get(val) is not None:
            by[tuple(r[k] for k in keys)].append(r[val])
    return {k: (np.mean(v), np.std(v), len(v)) for k, v in by.items()}


# 3 x 3, all splits
def protocol_tables(rows):
    m = mean_by(rows, ["dataset", "basis", "protocol"])
    tr = mean_by(rows, ["dataset", "basis", "protocol"], "train_rmse")
    for ds in ("mlsmall", "ml100k", "ml20m"):
        if not any(k[0] == ds for k in m):
            continue
        print(f"\n### {TITLES[ds]}, 3 x 3, test RMSE\n")
        print("| split | " + " | ".join(BASES) + " |")
        print("|---" * 7 + "|")
        for p in PROTO:
            cells = []
            vals = {b: m[(ds, b, p)][0] for b in BASES if (ds, b, p) in m}
            best = min(vals, key=vals.get) if vals else None
            for b in BASES:
                if b in vals:
                    s = f"{vals[b]:.4f}"
                    cells.append(f"**{s}**" if b == best else s)
                else:
                    cells.append("")
            print(f"| {PROTO[p]} | " + " | ".join(cells) + " |")
        p = "kfold" if (ds, "C1", "kfold") in tr else "random"
        print(f"| *train RMSE ({PROTO[p]})* | " + " | ".join(
            f"*{tr[(ds, b, p)][0]:.4f}*" if (ds, b, p) in tr else "" for b in BASES) + " |")


def plot_3x3(rows):
    m = mean_by(rows, ["dataset", "basis", "protocol"])
    tr = mean_by(rows, ["dataset", "basis", "protocol"], "train_rmse")
    dss = [d for d in ("mlsmall", "ml100k", "ml20m") if any(k[0] == d for k in m)]
    fig, axes = plt.subplots(1, len(dss), figsize=(3.4 * len(dss), 3.6), dpi=160)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, dss):
        style(ax)
        p = "kfold" if (ds, "C1", "kfold") in m else "random"
        for x, b in enumerate(BASES):
            if (ds, b, p) not in m:
                continue
            ax.plot(x, m[(ds, b, p)][0], marker=MARK[b], color=COLOR[b], markersize=8,
                    linestyle="none")
            if (ds, b, p) in tr:
                ax.plot(x, tr[(ds, b, p)][0], marker=MARK[b], markersize=8, linestyle="none",
                        markerfacecolor="none", markeredgecolor=COLOR[b], markeredgewidth=1.5)
                ax.plot([x, x], [tr[(ds, b, p)][0], m[(ds, b, p)][0]], color=GRID, linewidth=2,
                        zorder=0)
        ax.set_xticks(range(6), BASES)
        ax.set_title(f"{TITLES[ds]}\n{PROTO[p]}", fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("RMSE (filled = test, hollow = train)", color=MUTED, fontsize=8)
    fig.suptitle("Six co-clustering bases, 3 x 3, the gap is overfitting", fontsize=10,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    path = os.path.join(OUT, "figures", "bases_3x3.png")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# the grid
def plot_square(rows, ds):
    m = mean_by([r for r in rows if r["dataset"] == ds and r["k"] == r["l"]], ["basis", "k"])
    if not m:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4), dpi=160)
    style(ax)
    sizes = sorted({k for _, k in m})
    for b in BASES:
        ks = [k for k in sizes if (b, k) in m]
        if not ks:
            continue
        x = [sizes.index(k) for k in ks]
        y = [m[(b, k)][0] for k in ks]
        ax.plot(x, y, color=COLOR[b], marker=MARK[b], markersize=4, linewidth=2)
        ax.annotate(b, (x[-1], y[-1]), textcoords="offset points", xytext=(6, 0),
                    va="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(sizes)), [f"{k}x{k}" for k in sizes])
    ax.set_xlabel("K x L, only K = L", color=MUTED, fontsize=9)
    ax.set_ylabel("RMSE (lower is better)", color=MUTED, fontsize=9)
    p = rows[0]["protocol"] if rows else ""
    split = "5-fold CV mean" if any(r["dataset"] == ds and r["protocol"] == "kfold"
                                    for r in rows) else "random 80/20"
    ax.set_title(f"{TITLES[ds]}, six bases, {split}", fontsize=10, color=INK, loc="left")
    handles = [plt.Line2D([], [], color=COLOR[b], marker=MARK[b], linewidth=2, label=NAMES[b])
               for b in BASES]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    path = os.path.join(OUT, "figures", f"bases_square_{ds}.png")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def plot_heat(rows, ds):
    m = mean_by([r for r in rows if r["dataset"] == ds], ["basis", "k", "l"])
    ks = sorted({k for _, k, _ in m})
    ls = sorted({l for _, _, l in m})
    if len(m) < 6 * len(ks) * len(ls):
        return
    vals = np.array([v[0] for v in m.values()])
    lo, hi = vals.min(), np.percentile(vals, 95)          # one color scale for all six
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.8), dpi=150)
    for ax, b in zip(axes.flat, BASES):
        M = np.array([[m[(b, k, l)][0] for l in ls] for k in ks])
        ax.imshow(M, cmap="Blues_r", vmin=lo, vmax=hi, aspect="auto")
        best = np.unravel_index(np.argmin(M), M.shape)
        ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False,
                                   edgecolor="#eb6834", linewidth=2))
        ax.set_title(f"{NAMES[b]}\nbest {ks[best[0]]} x {ls[best[1]]}: {M[best]:.4f}",
                     fontsize=8, color=INK, loc="left")
        ax.set_xticks(range(len(ls)), [str(x) for x in ls], fontsize=6)
        ax.set_yticks(range(len(ks)), [str(x) for x in ks], fontsize=6)
        ax.tick_params(colors=MUTED)
    for ax in axes[1]:
        ax.set_xlabel("L (movie clusters)", fontsize=7, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("K (user clusters)", fontsize=7, color=MUTED)
    fig.suptitle(f"{TITLES[ds]}: RMSE for every K x L, 5-fold mean, darker = better, "
                 f"same color scale in all six", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    path = os.path.join(OUT, "figures", f"bases_heat_{ds}.png")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def grid_tables(rows):
    for ds in ("mlsmall", "ml100k", "ml20m"):
        sub = [r for r in rows if r["dataset"] == ds]
        if not sub:
            continue
        m = mean_by(sub, ["basis", "k", "l"])
        print(f"\n### {TITLES[ds]}, grid summary\n")
        print("| basis | 1 x 1 | 3 x 3 | 10 x 10 | 50 x 50 | best cell | best RMSE |")
        print("|---|---|---|---|---|---|---|")
        for b in BASES:
            cells = {(k, l): v for (bb, k, l), v in m.items() if bb == b}
            if not cells:
                continue
            best = min(cells, key=lambda kl: cells[kl][0])
            get = lambda k: f"{cells[(k, k)][0]:.4f}" if (k, k) in cells else ""  # noqa: E731
            print(f"| {NAMES[b]} | {get(1)} | {get(3)} | {get(10)} | {get(50)} | "
                  f"{best[0]} x {best[1]} | {cells[best][0]:.4f} |")


def best_tables(rows):
    """Each basis at its own best K x L, every split (run_bases.py best)."""
    rows = [r for r in rows if (r["k"], r["l"]) != (3, 3)]
    m = mean_by(rows, ["dataset", "basis", "protocol"])
    size = {(r["dataset"], r["basis"]): (r["k"], r["l"]) for r in rows}
    for ds in ("mlsmall", "ml100k", "ml20m"):
        if not any(k[0] == ds for k in m):
            continue
        print(f"\n### {TITLES[ds]}, each basis at its best K x L, test RMSE\n")
        heads = [f"{b} {size[(ds, b)][0]}x{size[(ds, b)][1]}" if (ds, b) in size else b
                 for b in BASES]
        print("| split | " + " | ".join(heads) + " |")
        print("|---" * 7 + "|")
        for p in PROTO:
            vals = {b: m[(ds, b, p)][0] for b in BASES if (ds, b, p) in m}
            best = min(vals, key=vals.get) if vals else None
            print(f"| {PROTO[p]} | " + " | ".join(
                ("" if b not in vals else (f"**{vals[b]:.4f}**" if b == best else f"{vals[b]:.4f}"))
                for b in BASES) + " |")


def main():
    prot = load("protocols.jsonl")
    three = [r for r in prot if (r["k"], r["l"]) == (3, 3)]
    grid = load("grid.jsonl")
    if three:
        plot_3x3(three)
        protocol_tables(three)
    best_tables(prot)
    for ds in ("mlsmall", "ml100k", "ml20m"):
        plot_square(grid, ds)
        plot_heat(grid, ds)
    grid_tables(grid)


if __name__ == "__main__":
    main()
