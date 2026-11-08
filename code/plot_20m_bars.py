"""The six bases on MovieLens 20M as a grouped bar chart.

    python3 plot_20m_bars.py

The sizes 3 x 3, 5 x 5, ... are separate settings, not points on a continuous
axis, so a line between them would be wrong. A grouped bar chart shows them as
separate groups. On 20M only square sizes (K = L) were run, and the figure says so.

Reads ../results/grid.jsonl (random 80/20, one run per cell).
Output: ../results/figures/bases_bars_ml20m.png
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tema  # noqa: E402
from tema import BASES, INK, MUTED, GRID, ROSE  # noqa: E402

COLOR, EDGE = tema.BASIS, tema.EDGE
tema.apply()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
NAMES = tema.BASIS_NAME
YMIN = 0.82


def main():
    m = {}
    for line in open(os.path.join(OUT, "grid.jsonl")):
        r = json.loads(line)
        if r["dataset"] == "ml20m" and r["protocol"] == "random" and r["k"] == r["l"]:
            m[(r["basis"], r["k"])] = r["rmse"]
    sizes = sorted({k for _, k in m})

    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=160)
    x = np.arange(len(sizes))
    w = 0.13
    for j, b in enumerate(BASES):
        y = [m[(b, k)] for k in sizes]
        ax.bar(x + (j - 2.5) * w, [v - YMIN for v in y], w, bottom=YMIN, color=COLOR[b], edgecolor=EDGE[b],
               linewidth=0.8, label=NAMES[b])
    # best basis of each size: a small mark above its bar
    for i, k in enumerate(sizes):
        j = int(np.argmin([m[(b, k)] for b in BASES]))
        ax.plot(x[i] + (j - 2.5) * w, m[(BASES[j], k)] + 0.008, marker="v", color=ROSE, markersize=5)

    ax.set_xticks(x, [f"{k} × {k}" for k in sizes])
    ax.set_xlabel("grid size K × L (user groups × movie groups). Only K = L was run on 20M", color=MUTED, fontsize=9)
    ax.set_ylabel(f"RMSE (lower is better, axis starts at {YMIN})", color=MUTED, fontsize=9)
    ax.set_ylim(YMIN, 1.06)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
    ax.set_title("MovieLens 20M, six co-clustering bases, random 80/20. ▼ = best basis of each size",
                 fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    path = os.path.join(OUT, "figures", "bases_bars_ml20m.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
