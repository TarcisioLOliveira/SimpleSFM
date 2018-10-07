#!/usr/bin/python2

import numpy as np
import cv2
import glob

BOARD_W = 6
BOARD_H = 6

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((BOARD_W*BOARD_H,3), np.float32)
objp[:,:2] = np.mgrid[0:BOARD_H,0:BOARD_W].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('camera/*.jpg')

#for fname in images:
if True:
    fname = images[0]
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (BOARD_W,BOARD_H), None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        
        cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        
        # Draw and display the corners
#        cv2.drawChessboardCorners(img, (BOARD_W,BOARD_H), corners, ret)
#        cv2.imshow('img',img)
#        cv2.waitKey(0)
#
#cv2.destroyAllWindows()

if len(objpoints) == len(imgpoints) and len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.savez("camera.npz", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
else:
    print("not enough points to calibrate")

