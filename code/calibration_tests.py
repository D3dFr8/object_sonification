import numpy as np
import cv2 as cv

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((13*9,3), np.float32)
#square size in mm
sq_size = 40
objp[:,:2] = np.mgrid[0:13,0:9].T.reshape(-1,2) * sq_size

results = np.load("calib_results.npy", allow_pickle=True)
ret = results.item()["ret"]
mtx = results.item()["camera matrix"]
dist = results.item()["distortion coeff"]
rvecs = results.item()["rotation vector"]
tvecs = results.item()["translation vector"]
objpoints = results.item()["obj points"]
imgpoints = results.item()["img points"]
valid_imgs = results.item()["valid imgs"]

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

#------------ calculate re-projection (training) error ------------#
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "re-projection error: {}".format(mean_error/len(objpoints)) )


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