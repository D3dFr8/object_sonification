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
import os

if __name__ == "__main__":
    #load category file
    file_path = 'categories.txt'
    with open(file_path, 'r') as file:
        categories = file.readlines()
    
    #load dimensions file
    file_path = 'dims.txt'
    with open(file_path, 'r') as file:
        dims = file.readlines()
        
    #init audio
    hrirSr = lH.getSamplingRate()
    print(hrirSr)
    #audio, sr = st.loadMP3('snap.mp3', hrirSr)

    #chirp = create_chirp(200, 400, 0.07, hrirSr, 0.5)
    parent_conn, child_conn = Pipe()
    p = Process(target=audio_process, args=(child_conn, hrirSr))
    p.start()

    #get dictionary with results from the calibration
    calib_results = np.load("calib_results.npy", allow_pickle=True)
    #resolution for calibration images was 4056 × 3040, so we take half of that
    cam_mtx = calib_results.item()["camera matrix"]*0.5
    cam_mtx[2,2] = 1.0
    distortion = calib_results.item()["distortion coeff"]
    cam_mtx_inv = np.linalg.inv(cam_mtx)
    
    path = os.path.abspath(os.getcwd())
    
    picam = Picamera2()
    model = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
    imx500 = IMX500(model)

    #configure camera with fps and resolution. A quarter of the OG resolution is used to ensure stable performance
    config = picam.create_preview_configuration(
        main={"size": (1014, 760), "format": "XRGB8888"},
        controls={"FrameDurationLimits": (33333, 33333)}
    )
    picam.configure(config)
    #start camera
    picam.start_preview(Preview.QTGL)
    picam.start()
    
    print("Camera started, precc Ctrl+C to stop")
    try:
        frame_n = 0
        while True:
            frame_n += 1
            picam.take_photo(f"{path}/imagesForSeg/frame.jpg")
            if frame_n == 30:
                frame_n = 0
            
        
    except KeyboardInterrupt:
        parent_conn.send(False)
        p.join(timeout=2)

        picam.stop()
        picam.close()