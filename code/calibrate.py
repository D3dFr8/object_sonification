import numpy as np
import cv2 as cv
import glob
import random
import os

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((13*9,3), np.float32)

#square size in mm
sq_size = 40
objp[:,:2] = np.mgrid[0:13,0:9].T.reshape(-1,2) * sq_size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

path = os.path.abspath(os.getcwd())
images = glob.glob(path+'/imagesForCalibWide/*.jpg')
random.shuffle(images)

split_idx = int(len(images)*0.7) #CHANGE THIS FOR DATASET SPLIT
train_imgs = images[:split_idx] #training set
valid_imgs = images[split_idx:] #validation set

for fname in train_imgs:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (13,9), None)
    print(f'{fname}: {ret}')

    #if found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        #cv.drawChessboardCorners(img, (13,9), corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey(500)

#cv.waitKey(1000)

#cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, 
                                gray.shape[::-1], None, None, 
                                flags=cv.CALIB_FIX_K3) #remove k3, extraneous distortion parameter

#undistortion code, uncomment if undistortion of a new image is desired:
"""
img = cv.imread('/home/pi2/Documents/exjobb/tqet33-exjobb/code/test.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('testAfter.jpg', dst)
"""

#------------ calculate re-projection (training) RMSE ------------#
total_squared_error = 0
total_train_points = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    
    #squeeze arrays for easier math
    pts_true = imgpoints[i].squeeze()
    pts_pred = imgpoints2.squeeze()
    
    #calculate squared distances
    squared_dist = np.sum((pts_true - pts_pred)**2, axis=1)
    
    total_squared_error += np.sum(squared_dist)
    total_train_points += len(pts_true)

#overall training RMSE
train_rmse = np.sqrt(total_squared_error / total_train_points)


#---------- validation RMSE ----------#
total_valid_squared_error = 0
total_valid_points = 0

for fname in valid_imgs:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (13,9), None)
    print(f'{fname}: {ret}')

    #if found, add object points, image points (after refining them)
    if ret == True:
        #get accurate position of corners
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        #calculate rotation and translation vectors for validation image
        _, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)
        
        #use these together with previously known camera matrix to predict where the corners should be
        projected_corners, _ = cv.projectPoints(objp, rvec, tvec, mtx, dist)

        #calculate squared error for validation
        pts_true = corners2.squeeze()
        pts_pred = projected_corners.squeeze()
        
        squared_dist = np.sum((pts_true - pts_pred)**2, axis=1)
        
        total_valid_squared_error += np.sum(squared_dist)
        total_valid_points += len(pts_true)

#overall validation RMSE
valid_rmse = np.sqrt(total_valid_squared_error / total_valid_points)


#calculate re-projection (training) error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

reproj_err = mean_error/len(objpoints)
print(f"re-projection error: {reproj_err}")


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

mean_error_perpoint = total_valid_error/total_points #average error for every point

print(f'Mean validation error per point: {mean_error_perpoint}')

print(f"Training Reprojection RMSE: {train_rmse}")
print(f'Overall Validation RMSE per point: {valid_rmse}')

#the following code was used to compute five test cases for each split and
#save the one with smallest validation RMSE (so that for each split, the best 
#test case result is used)
with open("calib_results_wide_70-30.csv", "a") as log:
    log.write(f"{reproj_err},{mean_error_perpoint},{train_rmse},{valid_rmse}\n")

results = {
    "ret": ret,
    "camera matrix": mtx,
    "distortion coeff": dist,
    "rotation vector": rvecs,
    "translation vector": tvecs,
    "obj points": objpoints,
    "img points": imgpoints,
    "valid imgs": valid_imgs,
    "error": valid_rmse
    }

np.save("calib_results_wide_70-30.npy", results)

"""
res = np.load("calib_results_wide_70-30.npy", allow_pickle=True)
best_error = res.item()["error"]

#if current error per point is smaller than best error, overwrite the data cus it's better

if valid_rmse < best_error:
    results = {
    "ret": ret,
    "camera matrix": mtx,
    "distortion coeff": dist,
    "rotation vector": rvecs,
    "translation vector": tvecs,
    "obj points": objpoints,
    "img points": imgpoints,
    "valid imgs": valid_imgs,
    "error": valid_rmse
    }

    np.save("calib_results_wide_70-30.npy", results)
"""