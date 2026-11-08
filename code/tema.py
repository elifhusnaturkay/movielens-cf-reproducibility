"""One color theme for the new figures (pistachio, lime and sage greens).

    import tema
    tema.apply()                      # once, before drawing
    ax.bar(..., color=tema.BASIS["C6"], edgecolor=tema.EDGE["C6"])
    tema.style(ax); tema.save(fig, path)

When more colors are needed (six bases, several categories), cream, dusty pink
and lavender are added, so that every group can be told apart. C6 always gets
the deepest green.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- the color card ---------------------------------------------------------
PISTACHIO = "#93c572"
LIME = "#c4dd7a"
SAGE = "#a3b899"
OLIVE = "#5f8a3a"      # deep green, for the most important series
FOREST = "#2f5233"     # darkest green, for the best end of a color scale
CREAM = "#ecd9a0"
PINK = "#e6abbd"
LAVENDER = "#bfaee0"
ROSE = "#d16f93"       # markers and highlights (best cell, best bar)

INK = "#2f3a2c"        # text
MUTED = "#6b7a65"      # axis labels, notes
GRID = "#e6ede0"       # grid lines
BG = "#f4f7ee"         # background of covers and slides
OTHER = "#dfe5d8"      # "other" / not highlighted points

# a darker edge of the same hue for every fill, so pastel bars stay readable
EDGES = {PISTACHIO: "#6f9a4f", LIME: "#8faa45", SAGE: "#73886a", OLIVE: "#3f6124",
         CREAM: "#c2a85e", PINK: "#c27891", LAVENDER: "#8c7bb5"}

# categorical order: use the first n colors for n groups
CATEGORICAL = [OLIVE, PINK, LAVENDER, LIME, CREAM, SAGE, PISTACHIO]

# the six co-clustering bases, the same everywhere
BASES = ["C1", "C2", "C3", "C4", "C5", "C6"]
BASIS = dict(zip(BASES, [SAGE, LAVENDER, LIME, CREAM, PINK, OLIVE]))
EDGE = {b: EDGES[c] for b, c in BASIS.items()}
BASIS_NAME = {"C1": "C1", "C2": "C2 (MSSRCC residue 1)", "C3": "C3", "C4": "C4",
              "C5": "C5 (George & Merugu)", "C6": "C6 (MSSRCC residue 2)"}

# continuous values: dark green = low (for RMSE: better), cream white = high
GREENS = LinearSegmentedColormap.from_list("tema_greens", [FOREST, OLIVE, PISTACHIO, "#cde4a3", "#f6f9ec"])
# continuous values where high is "more" (counts, mean rating): light to dark
GREENS_UP = GREENS.reversed()


def edge(color):
    return EDGES.get(color, INK)


def apply():
    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=CATEGORICAL),
        "legend.frameon": False, "image.cmap": "tema_greens",
        "savefig.facecolor": "white", "savefig.dpi": 170,
    })
    try:
        plt.colormaps.register(GREENS)
    except ValueError:
        pass                                # already registered


def style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)


def save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
