from picamera2.devices.imx500 import IMX500
from picamera2 import Picamera2, Preview
import numpy as np
import cv2 as cv
import time

#get dictionary with results from the calibration
calib_results = np.load("calib_results.npy", allow_pickle=True)

cam_mtx = calib_results.item()["camera matrix"]*0.5
cam_mtx[2,2] = 1.0
distortion = calib_results.item()["distortion coeff"]

breakpoint()
#cam_mtx_inv = np.linalg.inv(cam_mtx)

#initalization
picam = Picamera2()
model = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
imx500 = IMX500(model)

#configure camera with fps and resolution
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
        metadata = picam.capture_metadata()

        outputs = imx500.get_outputs(metadata)

        if outputs:
            boxes, scores, classes = outputs[0], outputs[1], outputs[2]
            
            threshold = 0.5
            for i in range(len(boxes)):
                if scores[i] > threshold:
                    # extract normalized coordinates, confidence and object type
                    x, y, w, h = boxes[i]
                    confidence = scores[i]
                    category = classes[i]

                    # get exact pixel location:
                    width, height = 2028, 1520

                    pixel_x = x * width
                    pixel_y = y * height
                    pixel_w = w * width
                    pixel_h = h * height

                    u = float(pixel_x + (pixel_w/2))
                    v = float(pixel_y + (pixel_h/2))

                    center_coords = np.array([[[u, v]]], dtype=np.float64)
                    vec_2d = cv.undistortPoints(center_coords, cam_mtx, distortion)
                    
                    vec_x = vec_2d[0,0,0]
                    vec_y = vec_2d[0,0,1]

                    camera_vec = np.array([vec_x, vec_y])
                    print(f'Object of class {category} found with confidence {confidence:.2f} at {(u, v)}')
                    print(f'Camera vector: {camera_vec}')
                    print("-------------------------------------------")
        time.sleep(0.01)

except KeyboardInterrupt:
    picam.stop()
    picam.close()
