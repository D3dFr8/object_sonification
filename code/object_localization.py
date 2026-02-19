from picamera2.devices.imx500 import IMX500
from picamera2 import Picamera2, Preview
import numpy as np
import cv2 as cv
import time
from nearestHrir import *
from pydub import AudioSegment
import point
import soundTools
import loadHrir
import getHRIR
import angle
import threading as th
import asyncio
from pubSub import *
from multiprocessing import Process, Pipe
"""
Logga i en fil och köra med olika filter
"""

from point import *

#get dictionary with results from the calibration
calib_results = np.load("calib_results.npy", allow_pickle=True)

#resolution for calibration images was 4056 × 3040, so we take half of that
cam_mtx = calib_results.item()["camera matrix"]*0.5
cam_mtx[2,2] = 1.0
distortion = calib_results.item()["distortion coeff"]

cam_mtx_inv = np.linalg.inv(cam_mtx)

#initalization
picam = Picamera2()
model = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
imx500 = IMX500(model)

#configure camera with fps and resolution. Half of the OG resolution is used to ensure stable performance
config = picam.create_preview_configuration(
    main={"size": (2028, 1520), "format": "XRGB8888"},
    controls={"FrameDurationLimits": (33333, 33333)}
)
picam.configure(config)

hrirSr = loadHrir.getSamplingRate()
audio, sr = soundTools.loadMP3('snap.mp3', hrirSr)

target = []
yL = SoundChannel()
yR = SoundChannel()
ready = [False]
running = [True]

#t1 = th.Thread(target = child, args=(target, audio, sr, yL, yR, ready, running, ))
#t1.start()

parent_conn, child_conn = Pipe()
child_conn.send([target, audio, sr, yL, yR, ready, running])
p = Process(target=child, args=(child_conn,))
p.start()

#start camera
picam.start_preview(Preview.QTGL)
picam.start()

print("Camera started, precc Ctrl+C to stop")
try:
    while True:
        #get metadata from camera
        metadata = picam.capture_metadata()

        #extract output tensor from metadata
        objects = imx500.get_outputs(metadata)

        if objects:
            #extract the boxes, confidence scores, and object types from detected objects
            boxes, scores, classes = objects[0], objects[1], objects[2]
            
            best_i = -1
            largest_w = 0
            #minimum confidence for an object to be considered
            threshold = 0.5
            for i in range(len(boxes)):
                if scores[i] > threshold and classes[i] == 0:
                    if boxes[i][2] > largest_w:
                        largest_w = boxes[i][2]
                        best_i = i

            if best_i >= 0:
                # extract normalized coordinates, confidence and object type
                # the camera is preset to portrait mode, so we extract the coords and dims this way
                y, x, h, w = boxes[best_i]
                confidence = scores[best_i]
                category = classes[best_i]

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
                #camera_vec[1] = -camera_vec[1]

                #convert cartesian coordinates to spherical and compute distance
                r = np.linalg.norm(camera_vec)
                
                #azimuth = 90 - (180/np.pi * np.arccos(camera_vec[1]/r))
                #elevation = 180/np.pi * np.arctan(camera_vec[0])
                azimuth = np.degrees(np.arctan2(camera_vec[0], 1)) * -1
                elevation = np.degrees(np.arctan2(camera_vec[1], 1)) * -1

                real_w_human = 0.6
                focal_len = cam_mtx[0,0]*2
                distance = (real_w_human*focal_len)/float(pixel_w)

                print(f'Object of class {category} found with confidence {confidence:.2f}')
                #print(f'Camera vector: {camera_vec}')
                #print(f"Object at X: {x:.2f}, Y: {y:.2f} (Width: {pixel_w:.2f})")
                print(f'Azimuth: {float(azimuth)}, Elevation: {float(elevation)}, Distance: {float(distance)}')


                #left, right = findNearestHRIR(createPointFromSph(azimuth, elevation, distance))

                #left_conv = np.convolve(left, left_channel)
                #right_conv = np.convolve(right, right_channel)
                
                #sounddevice.play()
                #print(f'Left: {left}, Right: {right}')
                #print(f'L channel: {left_conv}, R channel: {right_conv}')
                
                #------------------Play sound-------------------#

                targetAz = angle.createAngleFromDegrees(azimuth)
                targetEl = angle.createAngleFromDegrees(elevation)
                targetR = distance

                target_point = point.createPointFromSph(targetAz,targetEl,targetR)

                conn = parent_conn.recv()
                target, ready = conn[0], conn[5]
                if ready[0] == False:
                    #if not yL.empty():
                    #    print("Playing sound")
                    
                    target.append(target_point)
                    ready[0] = True
                    child_conn.send([target, audio, sr, yL, yR, ready, running])


                #hL,hR = getHRIR.getHrirAtTargetNN(targetPoint)

                #yL = soundTools.conv(audio,hL).tolist()
                #yR = soundTools.conv(audio,hR).tolist()
            
            print("-------------------------------------------")

        time.sleep(0.01)

except KeyboardInterrupt:
    running[0] = False
    #t1.join()
    p.join()
    picam.stop()
    picam.close()
