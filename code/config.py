"""Training numbers of the deep learning scripts, in one place.

dl.py (NCF, MLP, hybrid), rnn.py (GRU, LSTM) and bst.py (the two transformers)
read their settings from here, so no number is hidden inside the code. The
values are the same ones that were used for all results so far.
"""

SEED = 42                 # the transformers and GRU/LSTM also use 43 and 44

# optimizer: Adam, the same start for every network
LR = 1e-3
WEIGHT_DECAY = 1e-5

# early stopping on a validation part of the train set
VAL_FRACTION = 0.1        # random 10% (temporal split: the newest 10% of each user)
PATIENCE = 3              # stop after 3 epochs without improvement
MIN_DELTA = 1e-4          # smaller improvements do not count
EPOCHS_DL = 40            # at most, NCF / MLP / hybrid
EPOCHS_SEQ = 30           # at most, GRU / LSTM / transformers

# batch size: bigger on 20M (more than 1M train ratings), else it takes too long
BIG = 1_000_000
BATCH_DL = (1024, 8192)   # (small datasets, 20M)
BATCH_SEQ = (256, 2048)
