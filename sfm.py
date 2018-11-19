#!/usr/bin/python2
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pathlib2 import Path
import os
import sys

def undistort(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    # x,y,w,h = roi
    # dst = dst[y:y+h, x:x+w]
    return dst

def write_ply(coords, colors, filename):
    ply_header = (
                '''ply
                format ascii 1.0
                element vertex {vertex_count}
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                end_header
                '''
                )
    points = np.hstack([coords, colors])
    with open(filename, 'w') as outfile:
        #outfile.write(ply_header.format(vertex_count=len(coords)))
        np.savetxt(outfile, points, '%f %f %f %d %d %d')

def ORB_detect(img1, img2):
    # Initiate ORB detector
    orb = cv.ORB_create(WTA_K=4)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(cv.cvtColor(img1, cv.COLOR_RGB2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv.cvtColor(img2, cv.COLOR_RGB2GRAY), None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING2)#, crossCheck=True)
    # Match descriptors.
    matches = bf.knnMatch(des1, des2, k=2)
    # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)

    # ratio test, stackoverflow
    good = []
    for j,(m,n) in enumerate(matches):
        if m.distance < 0.90*n.distance:
            good.append(m)

    return (good, kp1, kp2)

def SIFT_detect(img1, img2):
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(cv.cvtColor(img1, cv.COLOR_RGB2GRAY),None)
    kp2, des2 = sift.detectAndCompute(cv.cvtColor(img2, cv.COLOR_RGB2GRAY),None)
    return kp1, des1, kp2, des2

def kNNMatch(kp1, des1, kp2, des2, lowes_thresh=0.75):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for j in range(len(matches))]

    # ratio test as per Lowe's paper
    good = []
    for j,(m,n) in enumerate(matches):
        if m.distance < lowes_thresh*n.distance:
            good.append(m)

    return good

def findDecomposedEssentialMatrix(p1, p2, K):
    # fundamental matrix and inliers
    F, mask = cv.findFundamentalMat(p1, p2, cv.FM_RANSAC, 3, 0.99)
    mask = mask.astype(bool).flatten()
    E = np.dot(K.T, np.dot(F, K))

    _, R, t, _ = cv.recoverPose(E, p1[mask], p2[mask], K)

    # w, u, vt = cv.SVDecomp(E)
    # W = np.array([[0, -1, 0],
    #               [1,  0, 0],
    #               [0,  0, 1]])
    # Winv = np.array([[ 0, 1, 0],
    #                  [-1, 0, 0],
    #                  [ 0, 0, 1]])
    # R = np.dot(u, np.dot(W, vt.T))
    # t = np.array([u[:, 2]]).T
    #
    # print('----------------')

    return R, t

def SfM(img_path_list, K, distCoeffs = 0, pointCloud=[], cameraPoses=[], MIN_MATCH_COUNT=10):
    img1 = cv.imread(img_path_list[0], 1) # trainImage
    P1 = np.column_stack([np.eye(3), np.zeros(3)])

    for i, img_path in enumerate(img_path_list[1:]):

        if not img_path.endswith('.jpg') and not img_path.endswith('.png'):
            continue

        print('Reading img: {}'.format(img_path))
        img2 = cv.imread(img_path, 1) # queryImage

        #cv.imwrite("test"+str(i)+".png", undistort(img1, mtx, dist))

        ''' Feature extraction / Matching '''
        kp1, des1, kp2, des2 = SIFT_detect(img1, img2)
        matches = kNNMatch(kp1, des1, kp2, des2)

        if len(matches)>=MIN_MATCH_COUNT:
            points1 = np.array([kp1[x.queryIdx].pt for x in matches])
            points2 = np.array([kp2[x.trainIdx].pt for x in matches])
            pts1_norm = cv.undistortPoints(np.expand_dims(points1, axis=1), cameraMatrix=K, distCoeffs=distCoeffs)
            pts2_norm = cv.undistortPoints(np.expand_dims(points2, axis=1), cameraMatrix=K, distCoeffs=distCoeffs)

            ''' Param Estimation '''
            # E = findEssentialMatrix(pts1_norm, pts2_norm, K)
            R, t = findDecomposedEssentialMatrix(points1, points2, K)
            P2 = np.hstack((R, t))

            ''' Triangulation '''
            points_4d_hom = cv.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
            # points_3d = cv.convertPointsFromHomogeneous(points_4d_hom.T).reshape(-1,3)
            points_4d = points_4d_hom / np.tile(points_4d_hom[-1, :], (4, 1))
            points_3d = points_4d[:3, :].T
            pointCloud.append(points_3d)

            ''' Point Cloud '''
            x_coords = [int(kp1[x.queryIdx].pt[0]) for x in matches]
            y_coords = [int(kp1[x.queryIdx].pt[1]) for x in matches]
            image_coords = np.column_stack([x_coords, y_coords])
            colors = img1[y_coords, x_coords, :]
            write_ply(points_3d, colors, "meshes/mesh{}.ply".format(i))

            '''Camera Pose from 2D3DMatch'''
            _, rvec, tvec, inliers = cv.solvePnPRansac(
                                points_3d.astype(np.float64),
                                image_coords.astype(np.float64),
                                K, distCoeffs=distCoeffs, flags=cv.SOLVEPNP_ITERATIVE)

            cam_rmat, _ = cv.Rodrigues(rvec)
            camera_pose = np.concatenate([R, tvec], axis=1)
            cameraPoses.append(camera_pose)

            img1 = np.copy(img2)
            P1 = np.copy(P2)
        else:
            print("Not enough matches: "+str(len(matches))+"/"+str(MIN_MATCH_COUNT))

if __name__  == '__main__':
    #camera_data = np.load("calibration_data.npz")
    #K = camera_data['intrinsic_matrix']
    #distCoeffs = camera_data['distCoeff']

    #camera_data = np.load("camera.npz")
    #K = camera_data['mtx']
    #distCoeffs = camera_data['dist']

    if not os.path.isdir("./meshes"):
        os.mkdir("meshes")

    K = np.array([[2759.48, 0,       1520.69],
                  [0,       2764.16, 1006.81],
                  [0,       0,       1]])

    # path = Path('./data/berlin/images')
    path = Path('./data/fountain-P11/images')
    img_path_list = sorted([str(x) for x in path.iterdir()])
    SfM(img_path_list, K)#, distCoeffs=distCoeffs)
    #cv.imwrite("test0.png", undistort(img2, mtx, dist))

    # f = 2500.0
    # width = 1024.0
    # height = 768.0
    # K = np.array([[f,0,width/2],
    #               [0,f,height/2],
    #               [0,0,1]])
