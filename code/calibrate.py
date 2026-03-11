import numpy as np
import cv2 as cv
import glob
import random
import os
import json

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((13*9,3), np.float32)

#square size in mm
sq_size = 40
objp[:,:2] = np.mgrid[0:13,0:9].T.reshape(-1,2) * sq_size

# Arrays to store object points and image points from all the images.
trainobjpoints = [] # 3d point in real world space
trainimgpoints = [] # 2d points in image plane.

#dirname = os.path.dirname(__file__)
#filename = os.path.join(dirname, '/images')

#print(dirname)
path = os.path.abspath(os.getcwd())
images = glob.glob(path+'../imagesForCalib/*.jpg')
random.shuffle(images)

split_idx = int(len(images)*0.8)
train_imgs = images[:split_idx] #training set
valid_imgs = images[split_idx:] #validation set
#print(images)

for fname in train_imgs:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (13,9), None)
    print(f'{fname}: {ret}')

    # If found, add object points, image points (after refining them)
    if ret == True:
        trainobjpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        trainimgpoints.append(corners2)

        # Draw and display the corners
        #cv.drawChessboardCorners(img, (13,9), corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey(500)

#cv.waitKey(1000)

#cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(trainobjpoints, trainimgpoints, 
                                gray.shape[::-1], None, None, 
                                flags=cv.CALIB_FIX_K3) #remove k3, extraneous distortion parameter

results = {
    "ret": ret,
    "camera matrix": mtx,
    "distortion coeff": dist,
    "rotation vector": rvecs,
    "translation vector": tvecs
}

#np.save("calib_results.npy", results) #only if we want
"""
img = cv.imread('/home/pi2/Documents/exjobb/tqet33-exjobb/code/test.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('imageAfter.png', dst)
"""

#calculate re-projection (training) error
mean_error = 0
for i in range(len(trainobjpoints)):
    trainimgpoints2, _ = cv.projectPoints(trainobjpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(trainimgpoints[i], trainimgpoints2, cv.NORM_L2)/len(trainimgpoints2)
    mean_error += error

print( "re-projection error: {}".format(mean_error/len(trainobjpoints)) )


#---------- validation ----------#
total_valid_error = 0
total_points = 0

for fname in valid_imgs:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (13,9), None)
    print(f'{fname}: {ret}')

    # If found, add object points, image points (after refining them)
    if ret == True:
        #get accurate position of corners
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        #calculate rotation and translation vectors for validation image
        _, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)
        #use these together with previously known camera matrix to predict where the corners should be
        projected_corners, _ = cv.projectPoints(objp, rvec, tvec, mtx, dist)

        #error between predicted position of corners and actual
        error = cv.norm(corners2, projected_corners, cv.NORM_L2)
        total_valid_error += error
        total_points += len(corners2)

mean_error_perimg = total_valid_error/len(valid_imgs) #sum of all errors in one image (on average)
mean_error_perpoint = total_valid_error/total_points #average error for every point

print(f'Mean validation error per image: {mean_error_perimg}')
print(f'Mean validation error per point: {mean_error_perpoint}')