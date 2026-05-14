import numpy as np

results = np.load("calib_results.npy", allow_pickle=True)
dist = results.item()["distortion coeff"]

results = np.load("fps_dataNN_human_30sec.npy", allow_pickle=True)
timestamp = results.item()["human_entry_time"] #32.162190198898315 sec
fps = results.item()["fps"] #31.30580277714987

results = np.load("fps_dataNN_human.npy", allow_pickle=True)
fps = results.item()["fps"] #31.31856423294784

results = np.load("fps_dataNN_nothing.npy", allow_pickle=True)
fps = results.item()["fps"] #31.39205410103174
