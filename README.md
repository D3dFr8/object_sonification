_Instructions_

**Startup**

1. Navigate to "tqet33-exjobb/code/" in a terminal window.
2. Create a virtual environment and activate it.
3. Run "python3 object_localization.py" for object detection with color segmentation. Add "nn" as an argument to run the object detection with MobileNet SSD instead.

**Configurations**

The following configurations refer to changes made within "object_localization.py" unless specified otherwise

To change:
1. which object is to be detected with the neural network - modify line 353. It detects only a person as default. See categories.txt for the index (line no.) that corresponds to a given object.

2. the image resolution for the color segmentation - modify lines 489 & 490. They downscale the image array for stability. Make sure to change the kernel size on line 517 accordingly.

3. which object is to be detected with color segmentation - define the real-world dimensions [m] of the object on line 56-57 and modify line 516 according to the color of the object. The color ranges are defined at the top of the file. The default object is a balloon inflated to its maximum size.

4. the calibration data set used by the object detection - modify line 291. Calibration data set splits from 60/40 to 90/10 are contained within "tqet33-exjobb/code/test_data/calibration". To create a new data set, open file "calibrate.py" and modify line 22 (image set), 25 (training set size), and 192 (new data set name) accordingly. Run the file.


_Links_

OpenCV: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

Raspberry Pi setup: https://www.raspberrypi.com/documentation/computers/getting-started.html 
Raspberry Pi AI camera: https://www.raspberrypi.com/documentation/accessories/ai-camera.html