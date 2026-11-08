"""Fix every random seed with one call.

    from reproducibility import set_all_seeds
    set_all_seeds(42)

Python's random, NumPy's global generator, PyTorch (CPU, CUDA and Apple MPS) and
PYTHONHASHSEED. Our scripts also make their own np.random.default_rng(seed) for
shuffling, which does not depend on the global one, so this call does not change
any old result. It only closes the gaps. We do not use a DataLoader, so there are
no worker processes to seed.

GPU runs can still differ a little between machines (MPS and CUDA kernels are
not fully deterministic). seed_check.py measures how much on this Mac.
"""

import os
import random

import numpy as np


def set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)                # CPU and every CUDA device
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
