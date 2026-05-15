import numpy as np

results = np.load("calib_results.npy", allow_pickle=True)
dist = results.item()["distortion coeff"]

"""
#-------------MobileNet fps data------------#
results = np.load("test_data/fps/fps_dataNN_human_30sec.npy", allow_pickle=True)
timestamp = results.item()["human_entry_time"] #32.162190198898315 sec
fps = results.item()["fps"] #31.30580277714987

results = np.load("test_data/fps/fps_dataNN_human.npy", allow_pickle=True)
fps = results.item()["fps"] #31.31856423294784

results = np.load("test_data/fps/fps_dataNN_nothing.npy", allow_pickle=True)
fps = results.item()["fps"] #31.39205410103174

#-------------ColSeg fps data------------#
results = np.load("test_data/fps/fps_dataCol_blueBalloon_30sec.npy", allow_pickle=True)
timestamp2 = results.item()["color_entry_time"] #31.949613571166992 sec
fps = results.item()["fps"] #31.51236899004513

results = np.load("test_data/fps/fps_dataCol_blueBalloon.npy", allow_pickle=True)
fps = results.item()["fps"] #31.521732959767654

results = np.load("test_data/fps/fps_dataCol_nothing.npy", allow_pickle=True)
fps = results.item()["fps"] #31.532772301093946



#-------------MobileNet fps data per second------------#
results = np.load("per_second_fps_dataNN_human_30sec.npy", allow_pickle=True)
timestamp = results.item()["human_entry_time"] #30.89207410812378 sec
fps = results.item()["fps"] #mean: 29.522412386428517

results = np.load("per_second_fps_dataNN_human.npy", allow_pickle=True)
fps = results.item()["fps"] #mean: 29.52185052973058

results = np.load("per_second_fps_dataNN_nothing.npy", allow_pickle=True)
fps = results.item()["fps"] #mean: 29.52183555766092


#-------------ColSeg fps data per second------------#
results = np.load("per_second_fps_dataCol_blueBalloon_30sec.npy", allow_pickle=True)
timestamp = results.item()["color_entry_time"] #31.32926321029663 sec
fps = results.item()["fps"] #mean: 29.52269891516191
print(timestamp)
print(np.mean(fps))

results = np.load("per_second_fps_dataCol_blueBalloon.npy", allow_pickle=True)
fps = results.item()["fps"] #mean: 29.529439685328096
print(np.mean(fps))

results = np.load("per_second_fps_dataCol_nothing.npy", allow_pickle=True)
fps = results.item()["fps"] #mean: 29.53490502248249
print(np.mean(fps))
"""

"""
#--------------MobileNet data 120 sec--------------#
results = np.load("per_sec_latency_dataNN_human_30sec.npy", allow_pickle=True)
timestamp = results.item()["human_entry_time"] #33.56586527824402 sec
latency = results.item()["latency"] #mean: 21.668856171355852  ms

max_l = max(latency)
print(max_l)
latency_max_i = latency.index(max_l)
print(max_l - latency[latency_max_i])

#print(timestamp)
human_entry_i = int(round(timestamp[0]))-1
#print(np.mean(latency[human_entry_i:]))
#halftime_i = round(((len(latency)-1)-human_entry_i)/2)
#print(np.mean(latency[human_entry_i:halftime_i]))
#print(np.mean(latency[halftime_i:len(latency)]))

results = np.load("per_second_fps_dataNN_human_30sec.npy", allow_pickle=True)
timestamp = results.item()["human_entry_time"] #30.100055932998657 sec
fps = results.item()["fps"] #mean: 29.770193573063107

print(fps[latency_max_i])
#print(timestamp)
human_entry_i = int(round(timestamp[0]))-1
#print(np.mean(fps))
#Latency is 29.552698135375977 ms at max. At this same point, fps is 29.967384074139964
#It takes the HRIR ~29.553 ms to calculate, while a frame is processed in ~33.370 ms. No targets are thrown out

for i in range(len(fps)):
    if latency[i] > fps[i]:
        print("bad")


#---------------MobileNet data 120 sec stress test-------------#
results = np.load("per_sec_latency_dataNN_human_30sec_check.npy", allow_pickle=True)
latency = results.item()["latency"] #mean: 21.974200010299683  ms
print(np.mean(latency))

results = np.load("per_second_fps_dataNN_human_30sec_check.npy", allow_pickle=True)
fps = results.item()["fps"] #mean: 29.20164711900892
print(np.mean(fps))

for i in range(len(fps)):
    if latency[i] > fps[i]:
        print("bad")
        print(i)
#bottleneck at indices 110, 111, 112 and 115.
"""

#--------------ColSeg latency data per second --------------#
results = np.load("per_sec_latency_dataCol_blueBalloon_30sec_check.npy", allow_pickle=True)
timestamp = results.item()["color_entry_time"] #34.599791049957275 sec
latency = results.item()["latency"] #mean: 21.36759648377868 ms

#print(timestamp)
human_entry_i = int(round(timestamp[0]))-1
print(np.mean(latency))
#halftime_i = round(((len(latency)-1)-human_entry_i)/2)
#print(np.mean(latency[human_entry_i:halftime_i]))
#print(np.mean(latency[halftime_i:len(latency)]))

results = np.load("per_second_fps_dataCol_blueBalloon_30sec_check.npy", allow_pickle=True)
timestamp = results.item()["color_entry_time"] #31.15688443183899 sec
fps = results.item()["fps"] #mean: 29.772201347246206

#print(fps[latency_max_i])
#print(timestamp)
human_entry_i = int(round(timestamp[0]))-1
print(np.mean(fps))

for i in range(len(fps)):
    if latency[i] > fps[i]:
        print("bad")
        print(i)

#---------------Colseg data 120 sec stress test-------------#
#1:
#latency mean: 20.303083247825747
#fps mean: 29.769731259738634
#no bottleneck
#2:
#latency mean: 
#fps mean: 
#