"""Does the number of groups change if we use only the train ratings?

    python3 train_check.py

viz.py used all ratings. Here we take the random 80/20 split, keep only the
train part, and compute the same silhouette scores on the first 50 PCA
components ("centered" version: user mean removed). No t-SNE, no UMAP, so it
takes seconds. The 20M sample is the same 5,000 users as in viz.py.

Output: ../results/viz_train_check.md
"""

import json
import os

import numpy as np

import viz

HERE = os.path.dirname(os.path.abspath(__file__))


def pca50_sil(X):
    from sklearn.decomposition import PCA
    dense = X.toarray()
    Y = PCA(n_components=min(50, min(dense.shape) - 1), random_state=viz.SEED).fit_transform(dense)
    sil = viz.silhouettes(Y)
    return viz.KS[int(np.argmax(sil))], max(sil)


def train_only(ds):
    if ds == "ml20m":
        raw = viz.cs.load_ml20m()
        train, _ = viz.cs.split(raw)
        keep = np.unique(viz.load_ratings("ml20m")[:, 0])    # the same 5,000 users as in viz.py
        return train[np.isin(train[:, 0], keep)][:, :3]
    raw = viz.cs.load_mlsmall() if ds == "mlsmall" else viz.cs.load().astype(float)
    train, _ = viz.cs.split(raw)
    return train[:, :3]


def main():
    old = {}
    for line in open(os.path.join(HERE, "..", "results", "viz.jsonl")):
        r = json.loads(line)
        if r.get("method") == "PCA-50" and r["fill"] == "centered":
            old[(r["ds"], r["side"])] = (r["best_k"], r["best_sil"])
    rows = ["| Dataset | Points | All ratings: best K (silhouette) | Train only: best K (silhouette) |",
            "|---|---|---|---|"]
    for ds in ["mlsmall", "ml100k", "ml20m"]:
        r = viz.filter_min(train_only(ds))
        A, _, _ = viz.build_matrix(r)
        C = viz.center_users(A)
        print(f"{ds}: train {len(r)} ratings, {A.shape}", flush=True)
        for side, X in [("users", C), ("movies", C.T.tocsr())]:
            k, s = pca50_sil(X)
            ok, os_ = old[(ds, side)]
            print(f"  {side}: all {ok} ({os_:.2f})  train {k} ({s:.2f})", flush=True)
            rows.append(f"| {viz.DS_NAME[ds]} | {viz.SIDE_NAME[side]} | {ok} ({os_:.2f}) | {k} ({s:.2f}) |")
    with open(os.path.join(HERE, "..", "results", "viz_train_check.md"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
