from picamera2.devices.imx500 import IMX500
from picamera2 import Picamera2, Preview
import numpy as np
import cv2 as cv
import time
from pydub import AudioSegment
import point
import soundTools
import loadHrir
import getHRIR
import angle
import threading as th
import asyncio
from multiprocessing import Process, Pipe
from point import *
import matplotlib.pyplot as plt
"""
Logga i en fil och köra med olika filter
"""

#----------------------------------------------#
#-----------------AUDIO HANDLER----------------#
#----------------------------------------------#
def audio_process(conn, audio, sr):
    while True:
        #returns True if there is data in pipeline. Block for 50 ms
        if conn.poll():
            try:
                #extract latest pipeline data
                msg = conn.recv()

                #check if we want to exit
                if msg == False:
                    print("Audio process exiting")
                    break
                
                #extract coords
                az, el, r = msg

                #drain pipeline of all older messages
                while conn.poll():
                    msg = conn.recv()
                    if msg == False: return
                    az, el, r = msg

                #create target point
                targetAz = angle.createAngleFromDegrees(az)
                targetEl = angle.createAngleFromDegrees(el)
                targetR = r
                target_point = point.createPointFromSph(targetAz,targetEl,targetR)

                #calculate HRIR, convolve, and play sound
                hL,hR = getHRIR.getHrirAtTarget(target_point, 0.1)
                yL = soundTools.conv(audio,hL).tolist()
                yR = soundTools.conv(audio,hR).tolist()

                soundTools.playSound(yL,yR,sr)

            except EOFError:
                print("End of file error, something unexpected happened")
                break

#--------------------------------------------------#
#------------------MAIN CAMERA LOOP----------------#
#--------------------------------------------------#
if __name__ == "__main__":
    #init audio
    hrirSr = loadHrir.getSamplingRate()
    audio, sr = soundTools.loadMP3('snap.mp3', hrirSr)

    a = len(audio)
    f = np.arange(0, a//2 + 1) * (sr / a)
    #fig = plt.figure()
    #plt.plot(f,np.abs(np.fft.fft(audio))[:1+a//2])
    #plt.show()
    #start parent and child process
    parent_conn, child_conn = Pipe()
    p = Process(target=audio_process, args=(child_conn, audio, sr))
    p.start()

    #get dictionary with results from the calibration
    calib_results = np.load("calib_results.npy", allow_pickle=True)
    #resolution for calibration images was 4056 × 3040, so we take half of that
    cam_mtx = calib_results.item()["camera matrix"]*0.5
    cam_mtx[2,2] = 1.0
    distortion = calib_results.item()["distortion coeff"]
    cam_mtx_inv = np.linalg.inv(cam_mtx)

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
                
                idx_widest = -1
                widest_box = 0
                #minimum confidence for an object to be considered
                threshold = 0.5
                for i in range(len(boxes)):
                    if scores[i] > threshold and classes[i] == 0:
                        if boxes[i][2] > widest_box:
                            widest_box = boxes[i][2]
                            idx_widest = i

                # extract normalized coordinates, confidence and object type
                # the camera is preset to portrait mode, so we extract the coords and dims this way
                y, x, h, w = boxes[idx_widest]
                confidence = scores[idx_widest]
                category = classes[idx_widest]

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

                #compute distance based on IRL width of object
                real_w_human = 0.5
                focal_len = cam_mtx[0,0]*2
                #print(f'focalx: {focal_len/2}, focaly: {cam_mtx[1,1]}')
                distance = (real_w_human*focal_len)/float(pixel_w)

                target = (float(azimuth), float(elevation), float(distance))

                print(f'Object of class {category} found with confidence {confidence:.2f}')
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
