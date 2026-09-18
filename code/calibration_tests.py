import numpy as np
import cv2 as cv

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((13*9,3), np.float32)

#square size in mm
sq_size = 40
objp[:,:2] = np.mgrid[0:13,0:9].T.reshape(-1,2) * sq_size

#load and extract results from a chosen calibration model
results = np.load("calib_results_90-10.npy", allow_pickle=True)
ret = results.item()["ret"]
mtx = results.item()["camera matrix"]
dist = results.item()["distortion coeff"]
rvecs = results.item()["rotation vector"]
tvecs = results.item()["translation vector"]
objpoints = results.item()["obj points"]
imgpoints = results.item()["img points"]
valid_imgs = results.item()["valid imgs"]

#prepare new image for undistortion
img = cv.imread('/home/pi2/Documents/exjobb/tqet33-exjobb/code/experiments/test_imgs/testLines.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('testLinesAfter_90-10.jpg', dst)

#------------ calculate re-projection (training) error ------------#
mean_error = 0
for i in range(len(objpoints)):
    #predict where corner points should be
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    #compute error between predicted and actual positions
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print(f're-projection error: {mean_error/len(objpoints)}')

#---------- calculate validation error ----------#
total_valid_error = 0
total_points = 0

for fname in valid_imgs:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (13,9), None)
    print(f'{fname}: {ret}')

    #ff found, add object points, image points (after refining them)
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

#average validation error for every point
mean_error_perpoint = total_valid_error/total_points 
print(f'Mean validation error per point: {mean_error_perpoint}')

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

        # Calculate squared error for validation
        pts_true = corners2.squeeze()
        pts_pred = projected_corners.squeeze()
        
        squared_dist = np.sum((pts_true - pts_pred)**2, axis=1)
        
        total_valid_squared_error += np.sum(squared_dist)
        total_valid_points += len(pts_true)

#overall validation RMSE
valid_rmse = np.sqrt(total_valid_squared_error / total_valid_points)

print(f"Training Reprojection RMSE: {train_rmse}")
print(f'Overall Validation RMSE per point: {valid_rmse}')