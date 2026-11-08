# Seed check: the same seed in two separate processes

Latest small, random 80/20 (fingerprint 1f1d4736e82d), seed 42, Apple M2 GPU (MPS).

| Network | Run 1 RMSE | Run 2 RMSE | Largest difference of a prediction | Old run (before config.py) |
|---|---|---|---|---|
| ncf | 0.861117 | 0.861117 | 1.43e-06 | 0.861117 |
| mlp | 0.840668 | 0.840668 | 0.00e+00 | 0.840668 |
| hybrid | 0.841636 | 0.841635 | 3.36e-04 | 0.841636 |
