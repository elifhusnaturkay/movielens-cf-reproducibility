# Every regressor with default settings, on the SVR features

5-fold mean RMSE (± std over the folds). Sorted by latest small. Clipped to the rating scale.

| Model | Latest small | 100k | Time per fold (s, latest small) |
|---|---|---|---|
| CatBoost | 0.8365 ± 0.0042 | 0.9316 ± 0.0053 | 5 |
| Hist gradient boosting | 0.8392 ± 0.0045 | 0.9318 ± 0.0049 | 1 |
| Gradient boosting | 0.8432 ± 0.0042 | 0.9351 ± 0.0050 | 10 |
| SVR (RBF) | 0.8475 ± 0.0048 | 0.9448 ± 0.0054 | 262 |
| MLP | 0.8481 ± 0.0042 | 0.9398 ± 0.0059 | 14 |
| Ridge | 0.8488 ± 0.0040 | 0.9396 ± 0.0052 | 0 |
| Linear regression | 0.8488 ± 0.0040 | 0.9396 ± 0.0052 | 0 |
| Bayesian ridge | 0.8488 ± 0.0040 | 0.9396 ± 0.0052 | 0 |
| Random forest | 0.8495 ± 0.0040 | 0.9455 ± 0.0057 | 22 |
| XGBoost | 0.8504 ± 0.0045 | 0.9451 ± 0.0067 | 0 |
| Extra trees | 0.8511 ± 0.0051 | 0.9525 ± 0.0057 | 12 |
| Huber | 0.8531 ± 0.0038 | 0.9437 ± 0.0053 | 0 |
| KNN regressor | 0.9133 ± 0.0063 | 1.0141 ± 0.0048 | 1 |
| AdaBoost | 0.9143 ± 0.0042 | 0.9849 ± 0.0092 | 2 |
| Lasso | 1.0425 ± 0.0035 | 1.1257 ± 0.0041 | 0 |
| Elastic net | 1.0425 ± 0.0035 | 1.1257 ± 0.0041 | 0 |
| Mean (baseline) | 1.0425 ± 0.0035 | 1.1257 ± 0.0041 | 0 |
| Decision tree | 1.2125 ± 0.0049 | 1.3466 ± 0.0046 | 1 |

Our own hand-set models on the same folds (from ensemble.jsonl):

| Model | Latest small | 100k |
|---|---|---|
| our SVR (C = 0.3, ε = 0.3) | 0.8451 ± 0.0040 | 0.9404 ± 0.0054 |
| our Random Forest (300 trees) | 0.8371 ± 0.0042 | 0.9333 ± 0.0053 |
| our XGBoost (400 trees, depth 6) | 0.8373 ± 0.0042 | 0.9318 ± 0.0053 |
| our CatBoost | 0.8359 ± 0.0045 | 0.9291 ± 0.0053 |
| our AdaBoost | 0.8773 ± 0.0035 | 0.9628 ± 0.0040 |
| KNN, item based | 0.8527 ± 0.0051 | 0.9157 ± 0.0055 |
| stacking with XGBoost (uses the collaborative models) | 0.8140 ± 0.0049 | 0.8979 ± 0.0059 |
