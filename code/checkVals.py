import numpy as np

results = np.load("calib_results.npy", allow_pickle=True)
dist = results.item()["distortion coeff"]

print(dist)