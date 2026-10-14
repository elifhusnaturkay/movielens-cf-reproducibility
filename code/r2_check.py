# numbers from the paper, Table 1 (Dataset 1)
rmse_xgb, r2_xgb = 1.0063, 0.0794
var = rmse_xgb**2 / (1 - r2_xgb)     # variance of the test ratings

paper = {"DT": (1.3846, 0.7427), "KNN": (1.1473, 0.1966),
         "SVD": (0.8739, 0.78), "MBT4R": (0.62, 0.91)}

for name, (rmse, r2_printed) in paper.items():
    r2_from_rmse = 1 - rmse**2 / var
    print(f"{name:6} R2 printed {r2_printed:5.2f}   R2 from RMSE {r2_from_rmse:5.2f}")
