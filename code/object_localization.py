from picamera2.devices.imx500 import IMX500
from picamera2 import Picamera2, Preview
import numpy as np
import cv2 as cv
import time
from nearestHrir import *
from pydub import AudioSegment
#import sounddevice
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

#get path to model and use model
#imx500.config(picam, model_file=model)

#start camera
picam.start_preview(Preview.QTGL)
picam.start()

time.sleep(3)
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
            
            #minimum confidence for an object to be considered
            threshold = 0.5
            for i in range(len(boxes)):
                if scores[i] > threshold:
                    # extract normalized coordinates, confidence and object type
                    x, y, w, h = boxes[i]
                    confidence = scores[i]
                    category = classes[i]
                    #labels = get_labels()
                    #get exact pixel location of object:
                    width, height = 2028, 1520

                    pixel_x = x * width
                    pixel_y = y * height
                    pixel_w = w * width
                    pixel_h = h * height

                    #label = f"{labels[int(category)]} ({confidence:.2f})"
                    #cv.putText(m.array, label, (x + 5, y + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    #cv.rectangle(m.array, (x, y), (x + w, y + h), (0, 0, 255, 0))

                    #get coordinates of middle of object
                    u = float(pixel_x + (pixel_w/2))
                    v = float(pixel_y + (pixel_h/2))
                    w = 1.0

                    center_coords = np.array([u, v, w], dtype=np.float64)
                    camera_vec = cam_mtx_inv.dot(center_coords)
                    #vec_2d = cv.undistortPoints(center_coords, cam_mtx, distortion)
                    
                    #vec_x = vec_2d[0,0,0]
                    #vec_y = vec_2d[0,0,1]

                    #camera_vec = np.array([vec_x, vec_y])

                    #convert cartesian coordinates to spherical
                    r = np.linalg.norm(camera_vec)
                    theta = 90 - (180/np.pi * np.arccos(-camera_vec[1]/r))
                    phi = 180/np.pi * np.arctan(camera_vec[0])

                    print(f'Object of class {category} found with confidence {confidence:.2f} at {(u, v, w)}')
                    print(f'Camera vector: {camera_vec}')
                    print(f'Angles: {(phi, theta)}')

                    left, right = findNearestHRIR(createPointFromSph(phi, theta, 1))
                    audio = AudioSegment.from_mp3("soundreality-finger-snap-sound-423220.mp3")
                    
                    #left_conv = np.convolve(left, audio)
                    #right_conv = np.convolve(right, audio)
                    
                    #sounddevice.play()
                    #print(f'Left: {left_conv}, Right: {right_conv}')
            print("-------------------------------------------")

        time.sleep(0.01)

except KeyboardInterrupt:
    picam.stop()
    picam.close()
