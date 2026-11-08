"""Same seed, two separate processes: are the numbers the same?

    python3 seed_check.py              # NCF, MLP, hybrid on latest-small, random 80/20
    python3 seed_check.py one <kind>   # (used inside: one run, prints the RMSE)

Every run starts as a new Python process, so nothing is shared in memory. Each
network is trained twice with seed 42 (config.py, set_all_seeds) and the two
test predictions are compared. The RMSE is also compared with the run in
../results/deep.jsonl, which was made before config.py existed.
Needs ../results/preds/preds_mlsmall_random.npz (base_preds.py).

Output: ../results/seed_check.md
"""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "preds")
KINDS = ["ncf", "mlp", "hybrid"]


def one(kind, run):
    sys.path.insert(0, HERE)
    from dl import train_one, rmse
    d = dict(np.load(os.path.join(OUT, "preds_mlsmall_random.npz")))
    pred, val, ep = train_one(kind, d)
    np.save(os.path.join(OUT, f"seedcheck_{kind}_{run}.npy"), pred)
    print(json.dumps(dict(kind=kind, run=run, rmse=rmse(pred, d["y_test"]), epochs=ep)))


def old_rmse(kind):
    v = [r["rmse"] for r in map(json.loads, open(os.path.join(RESULTS, "deep.jsonl")))
         if r.get("model") == kind and r["dataset"] == "mlsmall" and r["protocol"] == "random"]
    return v[0] if v else None


def main():
    lines = ["# Seed check: the same seed in two separate processes", "",
             "Latest small, random 80/20 (fingerprint 1f1d4736e82d), seed 42, Apple M2 GPU (MPS).", "",
             "| Network | Run 1 RMSE | Run 2 RMSE | Largest difference of a prediction | Old run (before config.py) |",
             "|---|---|---|---|---|"]
    for kind in KINDS:
        res = []
        for run in (1, 2):
            out = subprocess.run([sys.executable, __file__, "one", kind, str(run)],
                                 capture_output=True, text=True, check=True).stdout
            res.append(json.loads(out.strip().splitlines()[-1]))
        a, b = (np.load(os.path.join(OUT, f"seedcheck_{kind}_{r}.npy")) for r in (1, 2))
        diff = float(np.abs(a - b).max())
        old = old_rmse(kind)
        lines.append(f"| {kind} | {res[0]['rmse']:.6f} | {res[1]['rmse']:.6f} | {diff:.2e} | "
                     f"{old:.6f} |" if old else "")
        print(lines[-1], flush=True)
        for r in (1, 2):
            os.remove(os.path.join(OUT, f"seedcheck_{kind}_{r}.npy"))
    with open(os.path.join(RESULTS, "seed_check.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote seed_check.md")


if __name__ == "__main__":
    if sys.argv[1:2] == ["one"]:
        one(sys.argv[2], int(sys.argv[3]))
    else:
        main()
