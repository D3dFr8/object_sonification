import numpy as np
import csv


results = np.load("calib_results_70-30.npy", allow_pickle=True)
mtx = results.item()["camera matrix"]
dist = results.item()["distortion coeff"]
T = results.item()["translation vector"]
R = results.item()["rotation vector"]

valid = results.item()["valid imgs"]
print(mtx)
print(dist)
print(T)
print(R)


"""
with open("test_data/calibration/calib_results_wide_90-10.csv", mode='r') as file:
    reader = csv.reader(file)

    data = [[float(val) for val in row] for row in reader]
        
num_cols = len(data[0])
num_rows = len(data)
    
print("Mean of each column:")
for col in range(num_cols):
    col_sum = sum(row[col] for row in data)
    col_mean = col_sum / num_rows
    print(f"Column {col}: {col_mean}")
"""


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
#--------------MobileNet latency data 120 sec--------------#
results = np.load("per_sec_latency_dataNN_human_30sec.npy", allow_pickle=True)
timestamp = results.item()["color_entry_time"] #34.7148380279541 sec
latency = results.item()["latency"] #mean after 30 sec: 21.55130742544151   ms

max_l = max(latency)
#print(max_l)
latency_max_i = latency.index(max_l)
#print(max_l - latency[latency_max_i])

print(timestamp)
human_entry_i = int(round(timestamp[0]))-1
print(np.mean(latency[human_entry_i:]))
#halftime_i = round(((len(latency)-1)-human_entry_i)/2)
#print(np.mean(latency[human_entry_i:halftime_i]))
#print(np.mean(latency[halftime_i:len(latency)]))

results = np.load("per_second_fps_dataNN_human_30sec.npy", allow_pickle=True)
timestamp = results.item()["color_entry_time"] #34.71368145942688 sec
fps = results.item()["fps"] #mean after 30 sec: 30.01701603533912

#print(fps[latency_max_i])
print(timestamp)
human_entry_i = int(round(timestamp[0]))-1
print(np.mean(fps[human_entry_i:]))
#Latency is 29.552698135375977 ms at max. At this same point, fps is 29.967384074139964
#It takes the HRIR ~29.553 ms to calculate, while a frame is processed in ~33.370 ms. No targets are thrown out

for i in range(len(fps)):
    if latency[i] > fps[i]:
        print("bad")

"""

"""
#--------------ColSeg latency data 120 sec--------------#
results = np.load("per_sec_latency_dataCol_blueBalloon_30sec.npy", allow_pickle=True)
timestamp = results.item()["color_entry_time"] #33.17031669616699 sec
latency = results.item()["latency"] #mean after 30 sec:22.20129418647152  ms

#print(timestamp)
human_entry_i = int(round(timestamp[0]))-1
print(timestamp)
print(np.mean(latency[human_entry_i:]))
#halftime_i = round(((len(latency)-1)-human_entry_i)/2)
#print(np.mean(latency[human_entry_i:halftime_i]))
#print(np.mean(latency[halftime_i:len(latency)]))

results = np.load("per_second_fps_dataCol_blueBalloon_30sec.npy", allow_pickle=True)
timestamp = results.item()["color_entry_time"] #33.169421672821045 sec
fps = results.item()["fps"] #mean after 30 sec: 30.016412237047067

#print(fps[latency_max_i])
#print(timestamp)
human_entry_i = int(round(timestamp[0]))-1
print(timestamp)
print(np.mean(fps[human_entry_i:]))

for i in range(len(fps)):
    if latency[i] > fps[i]:
        print("bad")
        print(i)

"""

"""
#---------------MobileNet data 120 sec stress test-------------#
results = np.load("per_sec_latency_dataNN_human_30sec_check.npy", allow_pickle=True)
latency = results.item()["latency"] #mean: 21.974200010299683  ms
print(np.mean(latency))
dropped = results.item()["dropped_targets"]
print(dropped)

results = np.load("per_second_fps_dataNN_human_30sec_check.npy", allow_pickle=True)
fps = results.item()["fps"] #mean: 29.20164711900892
print(np.mean(fps))

for i in range(len(fps)):
    if latency[i] > 1000/fps[i]:
        print("bad")
        print(i)
#bottleneck at index115.


#---------------Colseg data 120 sec stress test-------------#
#1:
#latency mean: 20.303083247825747
#fps mean: 29.769731259738634
#no bottleneck
#2:
#latency mean: 
#fps mean: 
#
"""

"""
#---------------MobileNet system load data-------------#
results = np.load("fps_RAM_CPU_temp_dataNN_human_30sec.npy", allow_pickle=True)
timestamp = results.item()["human_entry_time"] #34.24895262718201 sec
cpu_data = results.item()["cpu"] #mean: 19.13395061728395 %
temp_data = results.item()["temp"] #mean: 77.29012345679013 C
rss = results.item()["ram_rss"] #mean: 622.573688271605 MB
ram = results.item()["ram_sys"] #mean: 12.014814814814816 %
fps = results.item()["fps"] #mean: 30.036293247110258

object_entry_i = int(round(timestamp[0]))-1
print(timestamp)
print(np.mean(cpu_data[object_entry_i:]))
print(np.mean(temp_data[object_entry_i:]))
print(np.mean(rss[object_entry_i:]))
print(np.mean(ram[object_entry_i:]))
print(np.mean(fps[object_entry_i:]))

results = np.load("fps_RAM_CPU_temp_dataNN_human.npy", allow_pickle=True)
cpu_data = results.item()["cpu"] #mean: 19.03530701754386 %
temp_data = results.item()["temp"] #mean: 75.02543859649123 C
rss = results.item()["ram_rss"] #mean: 622.4428453947369 MB
ram = results.item()["ram_sys"] #mean: 11.852631578947367 %
fps = results.item()["fps"] #mean: 29.778836377138603
print(np.mean(cpu_data))
print(np.mean(temp_data))
print(np.mean(rss))
print(np.mean(ram))
print(np.mean(fps))

results = np.load("fps_RAM_CPU_temp_dataNN_nothing.npy", allow_pickle=True)
cpu_data = results.item()["cpu"] #mean: 8.345175438596492 %
temp_data = results.item()["temp"] #mean: 71.93289473684212 C
rss = results.item()["ram_rss"] #mean: 622.429961622807 MB
ram = results.item()["ram_sys"] #mean: 11.574561403508772 %
fps = results.item()["fps"] #mean: 29.78292303775268
print(np.mean(cpu_data))
print(np.mean(temp_data))
print(np.mean(rss))
print(np.mean(ram))
print(np.mean(fps))

results = np.load("latency_dataNN_human.npy", allow_pickle=True)
latency = results.item()["latency"] #mean: 20.76022378329573 ms
print(np.mean(latency))
dropped = results.item()["dropped_targets"]
print(dropped)

results = np.load("latency_dataNN_human_30sec.npy", allow_pickle=True)
timestamp = results.item()["human_entry_time"] #34.250109910964966 s
latency = results.item()["latency"] #mean: 21.37698729832967  ms
print(timestamp)
object_entry_i = int(round(timestamp[0]))-1
print(np.mean(latency[object_entry_i:]))
dropped = results.item()["dropped_targets"]
print(dropped)

for i in range(len(fps)):
    if latency[i] > 1000/fps[i]:
        print("bad")
        print(i)

"""

"""
#---------------ColSeg system load data-------------#
results = np.load("fps_RAM_CPU_temp_dataCol_blueBalloon_30sec.npy", allow_pickle=True)
timestamp = results.item()["color_entry_time"] # sec
cpu_data = results.item()["cpu"] #mean:  %
temp_data = results.item()["temp"] #mean:  C
rss = results.item()["ram_rss"] #mean:  MB
ram = results.item()["ram_sys"] #mean:  %
fps = results.item()["fps"] #mean: 

object_entry_i = int(round(timestamp[0]))-1
print(timestamp)
print(np.mean(cpu_data[object_entry_i:]))
print(np.mean(temp_data[object_entry_i:]))
print(np.mean(rss[object_entry_i:]))
print(np.mean(ram[object_entry_i:]))
print(np.mean(fps[object_entry_i:]))

results = np.load("fps_RAM_CPU_temp_dataCol_blueBalloon.npy", allow_pickle=True)
cpu_data = results.item()["cpu"] #mean:  %
temp_data = results.item()["temp"] #mean:  C
rss = results.item()["ram_rss"] #mean:  MB
ram = results.item()["ram_sys"] #mean:  %
fps = results.item()["fps"] #mean: 
print(np.mean(cpu_data))
print(np.mean(temp_data))
print(np.mean(rss))
print(np.mean(ram))
print(np.mean(fps))

results = np.load("fps_RAM_CPU_temp_dataCol_nothing.npy", allow_pickle=True)
cpu_data = results.item()["cpu"] #mean:  %
temp_data = results.item()["temp"] #mean:  C
rss = results.item()["ram_rss"] #mean:  MB
ram = results.item()["ram_sys"] #mean:  %
fps = results.item()["fps"] #mean: 
print(np.mean(cpu_data))
print(np.mean(temp_data))
print(np.mean(rss))
print(np.mean(ram))
print(np.mean(fps))

print("---------------------------")
results = np.load("latency_dataCol_blueBalloon.npy", allow_pickle=True)
latency = results.item()["latency"] #mean:  ms
print(np.mean(latency))
dropped = results.item()["dropped_targets"]


results = np.load("latency_dataCol_blueBalloon_30sec.npy", allow_pickle=True)
timestamp = results.item()["color_entry_time"] # s
latency = results.item()["latency"] #mean:  ms
print(timestamp)
object_entry_i = int(round(timestamp[0]))-1
print(np.mean(latency[object_entry_i:]))
dropped = results.item()["dropped_targets"]


for i in range(len(fps)):
    if latency[i] > 1000/fps[i]:
        print("bad")
        print(i)
"""

"""
#---------------MobileNet system load 1h data-------------#
results = np.load("NN_1h_stress_data.npy", allow_pickle=True)
cpu_data = results.item()["cpu"] #mean:  %
temp_data = results.item()["temp"] #mean:  C
rss = results.item()["ram_rss"] #mean:  MB
ram = results.item()["ram_sys"] #mean:  %
fps = results.item()["fps"] #mean: 
print(np.mean(cpu_data))
print(np.mean(temp_data))
print(np.mean(rss))
print(np.mean(ram))
print(np.mean(fps))

results = np.load("NN_1h_latency_data.npy", allow_pickle=True)
latency = results.item()["latency"] #mean: ms
print(np.mean(latency))
dropped = results.item()["dropped_targets"]
"""

"""
#---------------ColSeg system load 1h data-------------#
results = np.load("Col_1h_stress_data.npy", allow_pickle=True)
cpu_data = results.item()["cpu"] #mean:  %
temp_data = results.item()["temp"] #mean:  C
rss = results.item()["ram_rss"] #mean:  MB
ram = results.item()["ram_sys"] #mean:  %
fps = results.item()["fps"] #mean: 
print(np.mean(cpu_data[:589]))
print(np.mean(temp_data[:589]))
print(np.mean(rss[:589]))
print(np.mean(ram[:589]))
print(np.mean(fps[:589]))

results = np.load("Col_1h_latency_data.npy", allow_pickle=True)
latency = results.item()["latency"] #mean: ms
print(np.mean(latency[:589]))
dropped1 = results.item()["dropped_targets"][:589]

dropped_no_zero = []
for i in range(len(dropped1)):
    if dropped1[i] != 0:
        dropped_no_zero.append(dropped1[i])

print(dropped_no_zero)
print(np.mean(dropped1))
print(np.sum(dropped1))
"""

"""
#---------------ColSeg system load 10min data-------------#
results = np.load("Col_10min_stress_10000pix_data.npy", allow_pickle=True)
cpu_data = results.item()["cpu"] #mean:  %
temp_data = results.item()["temp"] #mean:  C
rss = results.item()["ram_rss"] #mean:  MB
ram = results.item()["ram_sys"] #mean:  %
fps = results.item()["fps"] #mean: 
print(np.mean(cpu_data))
print(np.mean(temp_data))
print(np.mean(rss))
print(np.mean(ram))
print(np.mean(fps))

results = np.load("Col_10min_latency_10000pix_data.npy", allow_pickle=True)
latency = results.item()["latency"] #mean: ms
print(np.mean(latency))
dropped = results.item()["dropped_targets"]

dropped_no_zero = []
for i in range(len(dropped)):
    if dropped[i] != 0:
        dropped_no_zero.append(dropped[i])

print(dropped_no_zero)
print(np.mean(dropped)*60)
print(np.sum(dropped))
"""