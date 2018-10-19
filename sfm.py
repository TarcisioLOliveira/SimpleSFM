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

def findEssentialMatrix(p1,p2, K):
    F = cv.findFundamentalMat(p1, p2, cv.FM_RANSAC, 0.1, 0.99)
    return np.matmul(K.T, np.matmul(F[0], K))

def decomposeEssential(E):
    w, u, vt = cv.SVDecomp(E)
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    Winv = np.array([[ 0, 1, 0],
                     [-1, 0, 0],
                     [ 0, 0, 1]])
    R = np.matmul(u, np.matmul(W, vt.T))
    t = np.array([u[:, 2]]).T
    return R, t

def SfM(img_path_list, K, pointCloud=[], cameraPoses=[], MIN_MATCH_COUNT=10):
    img2 = cv.imread(img_path_list[0], 1) # queryImage
    P2 = np.column_stack([np.eye(3), np.zeros(3)])

    for i, img_path in enumerate(img_path_list[1:]):
        img1 = cv.imread(img_path, 1) # trainImage

        #cv.imwrite("test"+str(i)+".png", undistort(img1, mtx, dist))

        ''' Feature extraction / Matching '''
        kp1, des1, kp2, des2 = SIFT_detect(img1, img2)
        matches = kNNMatch(kp1, des1, kp2, des2)

        if len(matches)>=MIN_MATCH_COUNT:
            points1 = np.array([kp1[x.queryIdx].pt for x in matches])
            points2 = np.array([kp2[x.trainIdx].pt for x in matches])
            pts1_norm = cv.undistortPoints(np.expand_dims(points1, axis=1), cameraMatrix=K, distCoeffs=None)
            pts2_norm = cv.undistortPoints(np.expand_dims(points2, axis=1), cameraMatrix=K, distCoeffs=None)

            ''' Param Estimation '''
            E = findEssentialMatrix(pts1_norm, pts2_norm, K)
            R, t = decomposeEssential(E)
            P1 = np.concatenate((R, t), axis=1)

            ''' Triangulation '''
            point_4d_hom = cv.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
            # point_4d_hom = cv.triangulatePoints(P1, P2, np.expand_dims(points1, axis=1), np.expand_dims(points2, axis=1))
            point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
            point_3d = point_4d[:3, :].T
            pointCloud.append(point_3d)

            ''' Point Cloud '''
            x_coords = [int(kp1[x.queryIdx].pt[0]) for x in matches]
            y_coords = [int(kp1[x.queryIdx].pt[1]) for x in matches]
            image_coords = np.column_stack([x_coords, y_coords])
            colors = img1[y_coords, x_coords, :]
            write_ply(point_3d, colors, "meshes/mesh"+str(i-1)+".ply")

            '''Camera Pose from 2D3DMatch'''
            _, rvec, tvec, inliers = cv.solvePnPRansac(
                                point_3d.astype(np.float64),
                                image_coords.astype(np.float64),
                                K, distCoeffs=None, flags=cv.SOLVEPNP_ITERATIVE)

            cam_rmat, _ = cv.Rodrigues(rvec)
            camera_pose = np.concatenate([R, tvec], axis=1)
            cameraPoses.append(camera_pose)

            img2 = np.copy(img1)
            P2 = np.copy(P1)
        else:
            print("Not enough matches: "+str(len(matches))+"/"+str(MIN_MATCH_COUNT))

if __name__  == '__main__':
    # if len(sys.argv) > 1:
    #     path = Path(sys.argv[2])
    #     camera_data = np.load(sys.argv[1])
    #     if sys.argv[1] == "calibration_data.npz":
    #         mtx = camera_data['intrinsic_matrix']
    #         dist = camera_data['distCoeff']
    #     else:
    #         ret = camera_data['ret']
    #         K = camera_data['mtx']
    #         dist = camera_data['dist']
    #         rvecs = camera_data['rvecs']
    #     tvecs = camera_data['tvecs']
    #
    # else:
    #     sys.exit()

        # if not os.path.isdir("./meshes"):
        #     os.mkdir("meshes")

    f = 2500
    width = 1024
    heigth = 768
    K = np.array([[f,0,width/2],
                    [0,f,heigth/2],
                    [0,0,1]])

    path = Path('data/crazyhorse')
    img_path_list = sorted([str(x) for x in path.iterdir()])
    SfM(img_path_list, K)
    #cv.imwrite("test0.png", undistort(img2, mtx, dist))

