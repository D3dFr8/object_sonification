_Introduction_

The goal of this project was to develop a physical prototypical platform that implemented
directional perception for the sonification of objects in an image captured by an AI camera,
with the ultimate goal to increase the independence of blind people exercising sports. The
project also evaluated and compared two different object detection algorithms- one using
the MobileNet SSD neural network included with the AI Camera, and the other making use of 
traditional color segmentation methods. In the context of the project, their 
technical and practical performance were analyzed and compared with each other. 
The project was carried out in collaboration with a concurrent master student, who developed 
the sonification of an object vector into spatial sound.

The system employs the pinhole camera model to translate 2D pixel coordinates of an object into 
a 3D target vector consisting of an azimuth, elevation and distance. The vectors are then 
rendered into spatial audio with Head-Related Impulse Responses (HRIR), and a sound is played 
inside a pair of headphones that corresponds to the direction of said object in the real world, along with 
a frequency attenuation depending on the distance to it.

_Expected Hardware_

1. Raspberry Pi 5.
2. Innomaker DAC HAT on top of the Pi, where the headphones are attached through a headphone jack.
3. AI Camera, inserted using a mini 22-pin connector into one of the two camera connectors located on top of the Pi.
4. Display, keyboard and mouse to navigate through the terminal and run the script.
5. A 3D-printed casing and stand for the camera, to ease carrying and handling of the system.
(Not obligatory but highly recommended to ensure physical stability of the camera during runtime).

For more information on the individual hardware components, see _Links_ at the bottom of this README. 
Below is an image of the complete system.


<img width="801" height="789" alt="image" src="https://github.com/user-attachments/assets/7841aecd-13cd-4afd-a8f4-11862aa768fd" />




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
Raspberry Pi 5: https://www.raspberrypi.com/products/raspberry-pi-5/  

Innomaker DAC HAT: https://www.inno-maker.com/hifi-dac-hat-for-raspberry-pi/

Raspberry Pi AI camera: https://www.raspberrypi.com/documentation/accessories/ai-camera.html
