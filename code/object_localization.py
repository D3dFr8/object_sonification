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

"""
Framtiden: Logga i en fil och köra med olika filter
"""
#init chirp
sr = lH.getSamplingRate()
#chirp = Chirp(400, 400, 1/30, sr, 0.2)

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
def audio_process(conn, sr):

    #real time audio generator
    #frames is updated through the hardware interrupt performed by the DAC
    def callback(output, frames, time_info, status):
        #give the signal to stop playing sound if object has been gone for more than 0.5 seconds
        #print(state.tracking)
        if time.time() - state.time_since_last_detection > 0.8:
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
        vol = 0.3#min(0.4, 0.3/max(state.dist, 0.3))
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

    try:
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

                #calculate HRIR
                hL_new,hR_new = getHRIR.getHrirAtTarget(target_point, 0.1)

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


    finally:
        stream.stop()
        stream.close()

#--------------------------------------------------#
#------------------MAIN CAMERA LOOP----------------#
#--------------------------------------------------#
if __name__ == "__main__":
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

    parent_conn, child_conn = Pipe()
    p = Process(target=audio_process, args=(child_conn, sr))
    p.start()

    #get dictionary with results from the calibration
    calib_results = np.load("calib_results.npy", allow_pickle=True)
    #resolution for calibration images was 4056 × 3040, so we take half of that
    cam_mtx = calib_results.item()["camera matrix"]*0.5
    cam_mtx[2,2] = 1.0
    distortion = calib_results.item()["distortion coeff"]
    cam_mtx_inv = np.linalg.inv(cam_mtx)

    focal_lenx = cam_mtx[0,0]
    focal_leny = cam_mtx[1,1]
    focal_area = focal_lenx*focal_leny
        
    #init camera
    picam = Picamera2()
    model = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
    imx500 = IMX500(model)

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
    print("Camera started, precc Ctrl+C to stop")
    try:        
        if localization_type == "nn":
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
                        camera_vec[0] = -camera_vec[0]

                        #compute horizontal and vertical angles
                        azimuth = np.degrees(np.arctan2(camera_vec[0], 1)) * -1
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
                
                    print("-------------------------------------------")

                time.sleep(0.01)
                
        elif localization_type == "col":
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
            
            balloon_w = 0.20
            balloon_h = 0.20
            real_area = balloon_w*balloon_h
            
            #ellipsoider
            while True:
                img = picam.capture_array()
                
                #convert from BGR to HSV (hue, saturation, value) color space
                hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

                #create mask and kernel
                mask = cv.inRange(hsv, lower_pink, upper_pink)
                kernel = np.ones((13, 13), np.uint8)
                
                #clean up noise using morphology (erosion and dilation)
                mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
                mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

                #find contours of the parts corresponding to the color chosen
                contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                
                #minimum pixel area of object to consider (10x10 size)
                MIN_AREA = 400
                
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
                    if area > biggest_box:
                        biggest_box = area
                        idx_biggest = i
                
                #process only the largest box (nearest object)
                if idx_biggest >= 0:
                    x, y, w, h = cv.boundingRect(filtered_contours[idx_biggest])
                    
                    #get coordinates of middle of object
                    u = float(x + (w/2))
                    v = float(y + (h/2))
                    w_coord = 1.0
                    
                    #create array with pixel coordinates and convert to camera's coordinates
                    center_coords = np.array([u, v, w_coord], dtype=np.float64)
                    camera_vec = cam_mtx_inv.dot(center_coords)
                    
                    #compute horizontal and vertical angles
                    azimuth = np.degrees(np.arctan2(camera_vec[0], 1))
                    elevation = np.degrees(np.arctan2(camera_vec[1], 1)) * -1
                    
                    pixel_area = w*h

                    distance = np.sqrt((real_area*focal_area)/float(pixel_area))

                    #if real_area < 0.15:
                     #   chirp.set_len(2)

                    #if real_area < 0.09:
                     #   chirp.set_len(20)

                    #print(f"len: {chirp.get_len()}, area: {real_area}")

                    target = (float(azimuth), float(elevation), float(distance))

                    #print(f'An object of color pink was found!')
                    #print(f'Camera vector: {camera_vec}')
                    #print(f"Object at X: {x:.2f}, Y: {y:.2f} (Width: {pixel_w:.2f})")
                    print(f'Azimuth: {target[0]}, Elevation: {target[1]}, Distance: {target[2]}')
                    
                    #send target point to pipeline for audio
                    parent_conn.send(target)
                
                print("-------------------------------------------")
                    
                time.sleep(0.01)

    except KeyboardInterrupt:
        parent_conn.send(False)
        p.join(timeout=2)

        picam.stop()
        picam.close()
