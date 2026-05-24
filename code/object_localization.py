from picamera2.devices.imx500 import IMX500
from picamera2 import Picamera2, Preview
import numpy as np
import time
import datatypes.point as point
import sound.soundTools as st
import sound.loadHrir as lH
import sound.getHRIR as getHRIR
import datatypes.angle as angle
from multiprocessing import Process, Pipe
#from datatypes.point import *
import matplotlib.pyplot as plt
from datatypes.chirp import Chirp
import sys
import os
import cv2 as cv
import glob
from scipy.signal import lfilter
import sounddevice as sd
import psutil

"""
Framtiden: Logga i en fil och köra med olika filter
"""
#init chirp
sr = lH.getSamplingRate()
#chirp = Chirp(400, 400, 1/30, sr, 0.2)

lower_red1 = np.array([0, 50, 16])
upper_red1 = np.array([10, 255, 255])

lower_orange = np.array([15, 50, 16])
upper_orange = np.array([25, 255, 255])

lower_yellow = np.array([25, 50, 16])
upper_yellow = np.array([35, 255, 255])

lower_green = np.array([40, 50, 16])
upper_green = np.array([70, 255, 255])

lower_turqoise = np.array([70, 50, 16])
upper_turqoise = np.array([90, 255, 255])

lower_lightblue = np.array([90, 50, 16])
upper_lightblue = np.array([100, 255, 255])

lower_blue = np.array([100, 50, 16])
upper_blue = np.array([110, 255, 255])

lower_marine = np.array([110, 50, 16])
upper_marine = np.array([130, 255, 255])

lower_purple = np.array([130, 50, 16])
upper_purple = np.array([150, 255, 255])
            
lower_pink = np.array([150, 50, 16])
upper_pink = np.array([165, 255, 255])

lower_red2 = np.array([170, 50, 16])
upper_red2 = np.array([180, 255, 255])
            
balloon_w = 0.26
balloon_h = 0.40
real_area_color = balloon_w*balloon_h

sensor_size = 0.007857
#A class to keep track of the state of different key variables for the audio process
#such as HRIR and filter memory
class State:
    hL = np.zeros(512) #default HRIR (basically nothing)
    hL[0] = 1
    hR = np.zeros(512)
    hR[0] = 1
    dist = 1
    time_since_last_detection = time.time() #+ 99999.0
    tracking = False
    ziL = None #memory for gapless filter
    ziR = None
    phase = 0

state = State()

#----------------------------------------------#
#-----------------AUDIO HANDLER----------------#
#----------------------------------------------#
def audio_process(conn, sr, master_clock):

    #real time audio generator
    #frames is updated through the hardware interrupt performed by the DAC
    def callback(output, frames, time_info, status):
        #give the signal to stop playing sound if object has been gone for more than 0.5 seconds
        #print(state.tracking)
        if time.time() - state.time_since_last_detection > 0.5:
            state.tracking = False
        
        #if no object has been found, 
        if not state.tracking:
            output.fill(0)
            state.phase = 0
            return
        
        #initialize memory for gapless filter if HRIR length changes
        if state.ziL is None or len(state.ziL) != len(state.hL) - 1:
            state.ziL = np.zeros(len(state.hL) - 1)
            state.ziR = np.zeros(len(state.hR) - 1)

        #generate hum with sawtooth wave:
        t = np.arange(frames)/sr
        f = 100

        current_t = t+state.phase
        #reduce volume with distance. Do not change volume closer than 40 cm
        vol = 0.2#min(0.4, 0.3/max(state.dist, 0.3))
        hum = vol*2*(f*current_t - np.floor(f*current_t + 0.5))
        state.phase += (frames/sr)

        #create new gapless filter based on previous and update memory
        #this is to ensure that the sound plays smoothly and doesn't click at the end of every audio chunk
        yL, state.ziL = lfilter(state.hL, [1.0], hum, zi=state.ziL)
        yR, state.ziR = lfilter(state.hR, [1.0], hum, zi=state.ziR)

        #max_val = max(np.max(np.abs(yL)), np.max(np.abs(yR)))
        #if max_val > 0:
         #   yL = yL/max_val
          #  yR = yR/max_val
        
        #send to DAC HAT
        stereo = np.column_stack((yL, yR))
        output[:] = np.ascontiguousarray(stereo, dtype=np.float32)

    #start the stream out to the hardware
    stream = sd.OutputStream(
        samplerate=sr,
        channels=2,
        dtype='float32',
        callback=callback
    )
    stream.start()

    
    latency = []
    atTime = []
    object_time = []
    dropped = []
    
    try:
        latency_interval = 1 #to display the latency every x second
        counter = 0
        calc_t = 0
        start_time = time.time()
        dropped_targets = 0
        while True:
            #returns True if there is data in pipeline. Block for 50 ms
            if conn.poll(0.05):
                #extract latest pipeline data
                msg = conn.recv()

                #check if we want to exit
                if msg == False:
                    print("Audio process exiting")
                    break

                #drain pipeline of all older messages
                while conn.poll():
                    msg = conn.recv()
                    dropped_targets += 1
                    if msg == False:
                        stream.stop()
                        return

                #extract coords
                az, el, r = msg

                #create target point
                targetAz = angle.createAngleFromDegrees(az)
                targetEl = angle.createAngleFromDegrees(el)
                targetR = r
                target_point = point.createPointFromSph(targetAz,targetEl,targetR)

                if len(object_time) == 0:
                    object_time.append(time.time()-master_clock)

                #calculate HRIR
                start_t = time.time()
                hL_new,hR_new = getHRIR.getHrirAtTarget(target_point, 0.1)
                calc_t = time.time() - start_t

                #normalize HRIR filters
                energy_val = max(np.sum(np.abs(hL_new)), np.sum(np.abs(hR_new)))
                #max_val = max(np.max(np.abs(hL_new)), np.max(np.abs(hR_new)))
                if energy_val > 0:
                    hL_new = hL_new/energy_val
                    hR_new = hR_new/energy_val
                
                #update states
                state.hL = hL_new
                state.hR = hR_new
                state.dist = targetR
                state.time_since_last_detection = time.time()
                state.tracking = True

            """
            counter+=1
            current_time = time.time()
            if (current_time - start_time) >= latency_interval:

                #dropped targets and latency measurement:
                dropped.append(dropped_targets)

                latency_val = calc_t*1000
                latency.append(latency_val) #latency will be in milliseconds

                time_val = current_time-master_clock
                atTime.append(time_val)

                with open("Col_10min_latency_stress_10000pix_log.csv", "a") as log:
                    log.write(f"{time_val},{latency_val},{dropped_targets}\n")

                dropped_targets = 0
                counter = 0
                
                start_time = time.time()
            """
            


    finally:
        stream.stop()
        stream.close()

        """
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        fig.suptitle('Calculation of HRIR with ColSeg (blue) - 10 min with 10000 pixel area', fontsize=16)

        # 1. Plot latency
        axs[0].plot(atTime, latency, color='green')
        axs[0].set_ylabel('HRIR Latency (ms)')
        axs[0].grid(True, linestyle='--', alpha=0.6)

        # 2. Plot dropped targets
        axs[1].plot(atTime, dropped, color='orange')
        axs[1].set_ylabel('Dropped targets')
        axs[1].set_xlabel('Time (s)') # Only the bottom graph needs the X-axis label
        axs[1].grid(True, linestyle='--', alpha=0.6)

        # Adjust layout so the labels don't overlap
        plt.tight_layout()
        plt.subplots_adjust(top=0.93) # Leave room for the main title

        # Save
        plt.savefig("Col_latency_10min_stress_10000pix_subplots.png", bbox_inches='tight')

        results = {
            "time": atTime,
            "latency": latency,
            "color_entry_time": object_time,
            "dropped_targets": dropped
        }

        np.save("Col_10min_latency_10000pix_data.npy", results)
        """
        
#--------------------------------------------------#
#------------------MAIN CAMERA LOOP----------------#
#--------------------------------------------------#
if __name__ == "__main__":
    fps = []
    atTime = []
    object_time = []

    cpu_data = []
    ram_data_overall = []
    ram_data_process = []
    temp_data = []

    this_process = psutil.Process()

    localization_type = "col"
    if len(sys.argv) > 1:
        localization_type = sys.argv[1]

    path = os.path.abspath(os.getcwd())
    
    #load category file
    file_path = 'categories.txt'
    with open(file_path, 'r') as file:
        categories = file.readlines()
    
    #load dimensions file
    file_path = 'dims.txt'
    with open(file_path, 'r') as file:
        dims = file.readlines()

    master_clock = time.time()
    
    parent_conn, child_conn = Pipe()
    p = Process(target=audio_process, args=(child_conn, sr, master_clock))
    p.start()

    #get dictionary with results from the calibration
    calib_results = np.load("calib_results_70-30.npy", allow_pickle=True)
    #resolution for calibration images was 4056 × 3040, so we take half of that
    cam_mtx = calib_results.item()["camera matrix"]*0.5
    cam_mtx[2,2] = 1.0
    distortion = calib_results.item()["distortion coeff"]
    cam_mtx_inv = np.linalg.inv(cam_mtx)

    focal_lenx = cam_mtx[0,0]
    focal_leny = cam_mtx[1,1]
    focal_area = focal_lenx*focal_leny
    
    # --- TRACKING VARIABLES SETUP ---
    last_known_distance = 0.0
    frames_lost_count = 0
    is_tracking = False

    # How many consecutive frames the object must be missing to be considered "lost". 
    # At 30 FPS, 15 frames is 0.5 seconds.
    LOSS_THRESHOLD = 30

    #init camera
    picam = Picamera2()
    model = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
    imx500 = IMX500(model)

    print("Camera started, precc Ctrl+C to stop")
    try:        
        if localization_type == "nn":
            #configure camera with fps and resolution. Half of the OG resolution is used to ensure stable performance
            config = picam.create_preview_configuration(
                main={"size": (2028, 1520), "format": "XRGB8888"},
                controls={"FrameDurationLimits": (33333, 33333)}
            )
            picam.configure(config)
            #start camera
            picam.start_preview(Preview.QTGL)
            picam.start()
            time.sleep(3)

            fps_interval = 1 #to display the frame rate every x second
            counter = 0
            #base_time = time.time()
            start_time = time.time()
            while True:
                #get metadata from camera
                metadata = picam.capture_metadata()

                #extract output tensor from metadata
                objects = imx500.get_outputs(metadata)

                if objects:
                    #extract the boxes, confidence scores, and object types from detected objects
                    boxes, scores, classes = objects[0], objects[1], objects[2]
                    
                    idx_biggest = -1
                    biggest_box = 0
                    #minimum confidence for an object to be considered
                    threshold = 0.5
                    
                    
                    for i in range(len(boxes)):
                        if scores[i] > threshold and classes[i] == 0:
                            if len(object_time) == 0:
                                object_time.append(time.time()-master_clock)
                            area = boxes[i][2]*boxes[i][3]
                            if area > biggest_box:
                                biggest_box = area
                                idx_biggest = i
                    

                    if idx_biggest >= 0:
                        # extract normalized coordinates, confidence and object type
                        # the camera is preset to portrait mode, so we extract the coords and dims this way
                        y, x, h, w = boxes[idx_biggest]
                        confidence = scores[idx_biggest]
                        class_i = int(classes[idx_biggest])

                        #get exact pixel location of object:
                        width, height = 2028, 1520
                        pixel_x = x * width
                        pixel_y = y * height
                        pixel_w = w * width
                        pixel_h = h * height

                        #get coordinates of middle of object
                        u = float(pixel_x + (pixel_w/2))
                        v = float(pixel_y + (pixel_h/2))
                        w_coord = 1.0

                        #create array with pixel coordinates and convert to camera's coordinates
                        center_coords = np.array([u, v, w_coord], dtype=np.float64)
                        camera_vec = cam_mtx_inv.dot(center_coords)
                        #camera_vec[0] = -camera_vec[0]

                        #compute horizontal and vertical angles
                        azimuth = np.degrees(np.arctan2(camera_vec[0], 1))
                        elevation = np.degrees(np.arctan2(camera_vec[1], 1)) * -1
                        
                        #extract class name and dimensions from index
                        category = categories[class_i][:-1]
                        dimensions = dims[class_i]
                        real_w = 1
                        real_h = 1
                        if '-' not in dimensions:
                            real_w = float(dimensions[:4])
                            real_h = float(dimensions[5:9])
                            
                        #print(real_w)
                        #print(real_h)
                        #compute distance based on IRL area of object
                        pixel_area = pixel_w*pixel_h
                        real_area = real_w*real_h

                        distance = np.sqrt((real_area*focal_area)/float(pixel_area))

                        last_known_distance = distance  # 'd' is your calculated distance variable
                        frames_lost_count = 0    # Reset the lost counter because we see the target
                        
                        if not is_tracking:
                            print(f"Target Acquired! Tracking started at {distance:.2f} meters.")
                            is_tracking = True
                        img_area = width*height
                        #distance = np.sqrt(focal_area*real_area*img_area/float(pixel_area)/(sensor_size**2))
                        #if real_area < 0.5:
                        #   chirp.set_len(0.4)
                        
                        #if real_area < 0.15:
                         #   chirp.set_len(2)

                        #if real_area < 0.09:
                         #   chirp.set_len(20)

                        #print(f"len: {chirp.get_len()}, area: {real_area}")

                        target = (float(azimuth), float(elevation), float(distance))

                        print(f'Object of class {category} found with confidence {confidence:.2f}')
                        #print(f'Camera vector: {camera_vec}')
                        #print(f"Object at X: {x:.2f}, Y: {y:.2f} (Width: {pixel_w:.2f})")
                        print(f'Azimuth: {target[0]}, Elevation: {target[1]}, Distance: {target[2]}')
                        
                        #send target point to pipeline for audio
                        parent_conn.send(target)
                    else:
                        if is_tracking:
                            frames_lost_count += 1
                            
                            # 3. Check if it has been missing for too many frames
                            if frames_lost_count >= LOSS_THRESHOLD:
                                print(f"\n--- TARGET LOST ---")
                                print(f"Maximum Distance Recorded: {last_known_distance:.2f} meters")
                                print(f"-------------------\n")
                                
                                # Reset state so it waits for the next time you walk into frame
                                is_tracking = False 
                                
                                # Optional: Save to a file so you don't lose the data
                                with open("dropoff_distances.txt", "a") as f:
                                    f.write(f"Lost at: {last_known_distance:.2f} meters\n")
                
                print("-------------------------------------------")

                #time.sleep(0.01)
                
                """
                counter+=1
                current_time = time.time()
                if (current_time - start_time) >= fps_interval:
                    #CPU:
                    cpu_cores = psutil.cpu_percent(percpu=True)
                    overall_cpu = sum(cpu_cores) / len(cpu_cores)
                    
                    #RAM:
                    ram = psutil.virtual_memory().percent
                    #convert from bytes to MB for this specific process
                    process_ram_mb = this_process.memory_info().rss / (1024*1024)

                    #Temperature (only on raspi):
                    try:
                        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                            temp_c = int(f.read()) / 1000.0
                    except FileNotFoundError:
                        temp_c = 0.0

                    cpu_data.append(overall_cpu)
                    ram_data_overall.append(ram)
                    ram_data_process.append(process_ram_mb)
                    temp_data.append(temp_c)

                    #FPS measurement:
                    fps_val = counter / (current_time - start_time)
                    fps.append(fps_val)
                    counter = 0

                    time_val = current_time-master_clock
                    atTime.append(time_val)

                    with open("NN_1_hour_stress_test_log.csv", "a") as log:
                        log.write(f"{time_val},{overall_cpu},{process_ram_mb},{ram},{temp_c},{fps_val}\n")

                    if current_time-master_clock >= 3600:
                        raise KeyboardInterrupt
                    start_time = time.time()
                """
                

                #fps.append(1.0 / (current_time - start_time))
                #atTime.append(current_time-master_clock)
                #if current_time-master_clock >= 60:
                #    raise KeyboardInterrupt
                #print("FPS: ", 1.0 / (current_time - start_time)) # FPS = 1 / time to process loop
                
        elif localization_type == "col":
            scale = 4
            #configure camera with fps and resolution. An eighth of the OG resolution is used to ensure stable performance
            config = picam.create_preview_configuration(
                main={"size": (int(2028/scale), int(1520/scale)), "format": "XRGB8888"},
                controls={"FrameDurationLimits": (33333, 33333)}
            )
            picam.configure(config)
            #start camera
            picam.start_preview(Preview.QTGL)
            picam.start()
            time.sleep(3)

            fps_interval = 1 #to display the frame rate every x second
            counter = 0
            #master_clock = time.time()
            start_time = time.time()
            while True:
                img = picam.capture_array()

                #convert from BGR to HSV (hue, saturation, value) color space
                hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

                #create mask and kernel
                mask = cv.inRange(hsv, lower_blue, upper_blue)
                kernel = np.ones((5, 5), np.uint8)
                
                #clean up noise using morphology with a kernel. Opening (erosion, dilation) and Closing (dilation, erosion)
                mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
                mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

                #find contours of the parts corresponding to the color chosen
                contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                
                #minimum pixel area of object to consider (10x10 size)
                MIN_AREA = 5000 / (scale**2)
                
                #filter contours to only include those larger than 10x10 pixels
                filtered_contours = []
                for contour in contours:
                    if cv.contourArea(contour) > MIN_AREA:
                        filtered_contours.append(contour)
                
                #choose index of biggest bounding box
                idx_biggest = -1
                biggest_box = 0
                for i in range(len(filtered_contours)):
                    x, y, w, h = cv.boundingRect(filtered_contours[i])
                    area = w*h
                    if len(object_time) == 0:
                        object_time.append(time.time()-master_clock)
                    if area > biggest_box:
                        biggest_box = area
                        idx_biggest = i
                
                #process only the largest box (nearest object)
                if idx_biggest >= 0:
                    x, y, w, h = cv.boundingRect(filtered_contours[idx_biggest])
                    
                    #get coordinates of middle of object
                    u = float(x + (w/2)) * scale
                    v = float(y + (h/2)) * scale
                    w_coord = 1.0
                    
                    #create array with pixel coordinates and convert to camera's coordinates
                    center_coords = np.array([u, v, w_coord], dtype=np.float64)
                    camera_vec = cam_mtx_inv.dot(center_coords)
                    
                    #compute horizontal and vertical angles
                    azimuth = np.degrees(np.arctan2(camera_vec[0], 1))
                    elevation = np.degrees(np.arctan2(camera_vec[1], 1)) * -1
                    
                    pixel_area = w*h * (scale**2)

                    distance = np.sqrt((real_area_color*focal_area)/float(pixel_area))

                    last_known_distance = distance  # 'd' is your calculated distance variable
                    frames_lost_count = 0    # Reset the lost counter because we see the target
                        
                    if not is_tracking:
                        print(f"Target Acquired! Tracking started at {distance:.2f} meters.")
                        is_tracking = True
                    #img_area = width*height
                    #if real_area_color < 0.15:
                     #   chirp.set_len(2)

                    #if real_area_color < 0.09:
                     #   chirp.set_len(20)

                    #print(f"len: {chirp.get_len()}, area: {real_area_color}")

                    target = (float(azimuth), float(elevation), float(distance))

                    #print(f'An object of color pink was found!')
                    #print(f'Camera vector: {camera_vec}')
                    #print(f"Object at X: {x:.2f}, Y: {y:.2f} (Width: {pixel_w:.2f})")
                    print(f'Azimuth: {target[0]}, Elevation: {target[1]}, Distance: {target[2]}')
                    
                    #send target point to pipeline for audio
                    parent_conn.send(target)
                else:
                    if is_tracking:
                        frames_lost_count += 1
                            
                        # 3. Check if it has been missing for too many frames
                        if frames_lost_count >= LOSS_THRESHOLD:
                            print(f"\n--- TARGET LOST ---")
                            print(f"Maximum Distance Recorded: {last_known_distance:.2f} meters")
                            print(f"-------------------\n")
                                
                            # Reset state so it waits for the next time you walk into frame
                            is_tracking = False 
                                
                            # Optional: Save to a file so you don't lose the data
                            with open("dropoff_distances.txt", "a") as f:
                                f.write(f"Lost at: {last_known_distance:.2f} meters\n")

                print("-------------------------------------------")

                
                
                #time.sleep(0.01)
                """
                counter+=1
                current_time = time.time()
                if (current_time - start_time) >= fps_interval:
                    #CPU:
                    cpu_cores = psutil.cpu_percent(percpu=True)
                    overall_cpu = sum(cpu_cores) / len(cpu_cores)
                    
                    #RAM:
                    ram = psutil.virtual_memory().percent
                    #convert from bytes to MB for this specific process
                    process_ram_mb = this_process.memory_info().rss / (1024*1024)

                    #Temperature (only on raspi):
                    try:
                        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                            temp_c = int(f.read()) / 1000.0
                    except FileNotFoundError:
                        temp_c = 0.0

                    cpu_data.append(overall_cpu)
                    ram_data_overall.append(ram)
                    ram_data_process.append(process_ram_mb)
                    temp_data.append(temp_c)

                    #FPS measurement:
                    fps_val = counter / (current_time - start_time)
                    fps.append(fps_val)
                    counter = 0

                    time_val = current_time-master_clock
                    atTime.append(time_val)

                    with open("Col_10min_stress_10000pix_log.csv", "a") as log:
                        log.write(f"{time_val},{overall_cpu},{process_ram_mb},{ram},{temp_c},{fps_val}\n")

                    if current_time-master_clock >= 600:
                        raise KeyboardInterrupt
                    start_time = time.time()
                """


    except KeyboardInterrupt:
        parent_conn.send(False)
        p.join(timeout=2)

        picam.stop()
        picam.close()
        
        #fps tests
        """
        plt.plot(atTime, fps)
        plt.title('Performance of MobileNet SSD per second with human at ~30 sec')
        plt.xlabel('Time (s)')
        plt.ylabel('FPS (1/s)')
        
        plt.savefig("per_second_fps_plotNN_human_30sec.png", bbox_inches='tight')

        results = {
            "time": atTime,
            "fps": fps,
            "color_entry_time": object_time
        }

        np.save("per_second_fps_dataNN_human_30sec.npy", results)
        """

        """
        #CPU, RAM, fps, and temperature tests
        fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
        fig.suptitle('Hardware load of ColSeg (blue) - 10min with 10000 pixel area', fontsize=16)

        # 1. Plot CPU
        axs[0].plot(atTime, cpu_data, color='green')
        axs[0].set_ylabel('CPU Usage (%)')
        axs[0].grid(True, linestyle='--', alpha=0.6)

        # 2. Plot Temperature
        axs[1].plot(atTime, temp_data, color='red')
        axs[1].set_ylabel('Temperature (°C)')
        # Optional: Add a dashed line showing the Pi's thermal throttle limit (usually ~80-85C)
        axs[1].axhline(y=80, color='black', linestyle='--', label='Throttle Limit') 
        axs[1].grid(True, linestyle='--', alpha=0.6)

        # 3. Plot RAM (RSS)
        axs[2].plot(atTime, ram_data_process, color='purple')
        axs[2].set_ylabel('RAM RSS (MB)')
        axs[2].grid(True, linestyle='--', alpha=0.6)

        # 4. Plot overall system RAM
        axs[3].plot(atTime, ram_data_overall, color='brown')
        axs[3].set_ylabel('system RAM (%)')
        axs[3].grid(True, linestyle='--', alpha=0.6)

        # 5. Plot FPS
        axs[4].plot(atTime, fps, color='blue')
        axs[4].set_ylabel('FPS')
        axs[4].set_xlabel('Time (s)') # Only the bottom graph needs the X-axis label
        axs[4].grid(True, linestyle='--', alpha=0.6)

        # Adjust layout so the labels don't overlap
        plt.tight_layout()
        plt.subplots_adjust(top=0.93) # Leave room for the main title

        # Save
        plt.savefig("Col_10min_stress_10000pix_subplots.png", bbox_inches='tight')
        plt.show()

        results = {
            "time": atTime,
            "fps": fps,
            "cpu": cpu_data,
            "temp": temp_data,
            "ram_rss": ram_data_process,
            "ram_sys":ram_data_overall,
            "color_entry_time": object_time
        }

        np.save("Col_10min_stress_10000pix_data.npy", results)
        """