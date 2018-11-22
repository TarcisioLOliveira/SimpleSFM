#!/usr/bin/python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pathlib2 import Path
import os
import sys
from bundle_adj import adjust

class SFM(object):
    def __init__(self, instrinsic, images_path):
        self.distCoeffs = 0
        self.MIN_MATCH_COUNT=10
        self.K = instrinsic

        img_path_list = sorted([str(x) for x in path.iterdir()])
        self.img_data = self.read_and_compute_keypoints(img_path_list)
        self.point_cloud = self.compute_initial_cloud(self.img_data[0], self.img_data[1])
        self.imgs_used = 2
        n_cameras = 1

        for img in self.img_data[2:]:
            camera_pose = self.estimate_new_view_pose(img)
            prev_img_idx = self.imgs_used - 1

            points1, points2, matches = self.kNNMatch(self.img_data[prev_img_idx], img)
            points_3d = self.triangulatePoints(self.img_data[prev_img_idx]['pose'], camera_pose,
                                               points1, points2)

            points_idx = [x.queryIdx for x in matches]
            self.img_data[self.imgs_used]['pose'] = camera_pose
            colors = self.get_point_colors_from_img(img, points1)

            camera_idx = np.full((len(points_3d)), 0, dtype=int)
            res = adjust(self.K, points_3d, n_cameras, len(points_3d), camera_idx, np.array(points_idx), points2)
            res = res.reshape((2, len(res)/2))
            points_3d = np.array((res[0, :], points_3d[:, 1], res[1, :])).T
            print points_3d

            point_cloud_data = {'points': points_3d,
                                'point_img_corresp': points_idx,
                                'colors': colors}

            self.point_cloud.append(point_cloud_data)
            self.imgs_used += 1


        self.write_ply(self.point_cloud)

    def read_and_compute_keypoints(self, img_path_list):
        img_data = []
        for img_path in img_path_list:
            print('Reading image and running SIFT: {}'.format(img_path))
            img = cv.imread(img_path, 1)
            kps, desc = self.SIFT_detect(img)
            img_name = img_path.split('/')[-1]
            img_data.append({'pixels': img, 'descriptors': desc, 'keypoints': kps})

        return img_data

    def SIFT_detect(self, img):
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(cv.cvtColor(img, cv.COLOR_RGB2GRAY),None)
        return kp, des

    def trainFlannMatch(self, img, current_descriptors, lowes_thresh=0.75):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv.FlannBasedMatcher(index_params, search_params)

        # each element is a set of descriptors from an image
        flann.add(current_descriptors)
        flann.train()

        # for each descriptor in the query, find the closest match
        matches = flann.match(queryDescriptors=img['descriptors'])
        matches = sorted(matches, key= lambda x:x.distance)
        return matches[:30]

    def kNNMatch(self, img1, img2, lowes_thresh=0.75):
        kp1, des1 = img1['keypoints'], img1['descriptors']
        kp2, des2 = img2['keypoints'], img2['descriptors']

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

        if len(good)>=self.MIN_MATCH_COUNT:
            points1 = np.array([kp1[x.queryIdx].pt for x in good])
            points2 = np.array([kp2[x.trainIdx].pt for x in good])
            return points1, points2, good

        return [], [], good

    def findDecomposedEssentialMatrix(self, p1, p2):
        # fundamental matrix and inliers
        F, mask = cv.findFundamentalMat(p1, p2, cv.FM_RANSAC, 3, 0.99)
        mask = mask.astype(bool).flatten()
        E = np.dot(self.K.T, np.dot(F, self.K))

        _, R, t, _ = cv.recoverPose(E, p1[mask], p2[mask], self.K)

        return R, t

    def triangulatePoints(self, P1, P2, points1, points2):
        pts1_norm = cv.undistortPoints(np.expand_dims(points1, axis=1),
                cameraMatrix=self.K, distCoeffs=self.distCoeffs)
        pts2_norm = cv.undistortPoints(np.expand_dims(points2, axis=1),
                cameraMatrix=self.K, distCoeffs=self.distCoeffs)
        points_4d_hom = cv.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
        # points_3d = cv.convertPointsFromHomogeneous(points_4d_hom.T).reshape(-1,3)
        points_4d = points_4d_hom / np.tile(points_4d_hom[-1, :], (4, 1))
        points_3d = points_4d[:3, :].T
        return points_3d

    def compute_initial_cloud(self, img1, img2):
            ''' Keypoint Matching '''
            points1, points2, matches = self.kNNMatch(img1, img2)

            if len(points1) == 0:
                print("Not enough matches: "+str(len(matches))+"/"+str(self.MIN_MATCH_COUNT))
                return None

            ''' Param Estimation '''
            R, t = self.findDecomposedEssentialMatrix(points1, points2)

            P1 = np.column_stack([np.eye(3), np.zeros(3)])
            P2 = np.hstack((R, t))

            ''' Triangulation '''
            points_3d = self.triangulatePoints(P1, P2, points1, points2)
            # ids of the matches used
            points_idx = [x.queryIdx for x in matches]

            self.img_data[0]['pose'] = P1
            self.img_data[1]['pose'] = P2

            colors = self.get_point_colors_from_img(img1, points1)

            point_cloud_data = {'points': points_3d,
                                'point_img_corresp': points_idx,
                                'colors': colors}

            return [point_cloud_data]

    def estimate_new_view_pose(self, img):
        print('New pose estimation')
        descriptors = [uImg['descriptors']
                       for uImg in self.img_data[:self.imgs_used]]
        # for uImg in self.img_data[:self.imgs_used]:
        #     descriptors.append(uImg['descriptors'])

        matches = self.trainFlannMatch(img, descriptors)

        # 3d Points
        points_3d = []
        points_2d = []
        for m in matches:
            # clouds are made of image pairs so (0,1) -> cloud_idx:0 (1,2) -> cloud_idx:1, ...
            cloud_idx = 0
            if m.imgIdx != 0:
                cloud_idx = m.imgIdx-1

            pointIdx = np.searchsorted(self.point_cloud[cloud_idx]['point_img_corresp'], m.trainIdx)
            if pointIdx == len(self.point_cloud[cloud_idx]['point_img_corresp']):
                continue

            # Get the 3d Point corresponding to the train image keypoint
            points_3d.append(self.point_cloud[cloud_idx]['points'][pointIdx])

            # 2d Points
            x_coords = int(img['keypoints'][m.queryIdx].pt[0])
            y_coords = int(img['keypoints'][m.queryIdx].pt[1])
            points_2d.append([x_coords, y_coords])


        # estimate camera pose from 3d2d Correspondences
        _, rvec, tvec, inliers = cv.solvePnPRansac(
                            np.array(points_3d, dtype=np.float64),
                            np.array(points_2d, dtype=np.float64),
                            self.K, None, confidence=0.99,
                            flags=cv.SOLVEPNP_ITERATIVE,
                            reprojectionError=8.)

        cam_rmat, _ = cv.Rodrigues(rvec)
        camera_pose = np.concatenate([cam_rmat, tvec], axis=1)
        return camera_pose


    def get_point_colors_from_img(self, img, matches):
        # x_coords = [int(img['keypoints'][x.queryIdx].pt[0]) for x in matches]
        # y_coords = [int(img['keypoints'][x.queryIdx].pt[1]) for x in matches]
        x_coords = [int(x[0]) for x in matches]
        y_coords = [int(x[1]) for x in matches]
        image_coords = np.column_stack([x_coords, y_coords])
        colors = img['pixels'][y_coords, x_coords, :]
        return colors


    def write_ply(self, point_cloud):
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
        for i, pc in enumerate(point_cloud):
            coords = pc['points']
            colors = pc['colors']
            filename = 'meshes/mesh{}.ply'.format(i)

            points = np.hstack([coords, colors])
            with open(filename, 'w') as outfile:
                #outfile.write(ply_header.format(vertex_count=len(coords)))
                np.savetxt(outfile, points, '%f %f %f %d %d %d')

if __name__  == '__main__':

    if not os.path.isdir("./meshes"):
        os.mkdir("meshes")

    # K = np.array([[2759.48, 0,       1520.69],
    #               [0,       2764.16, 1006.81],
    #               [0,       0,       1]])

    # path = Path('./data/fountain-P11/images')

    K = np.array([[3140.63, 0, 1631.5],
                  [0, 3140.63, 1223.5],
                  [0, 0, 1]])

    # f = 2500.0
    # width = 1024.0
    # height = 768.0
    # K = np.array([[f,0,width/2],
    #               [0,f,height/2],
    #               [0,0,1]])
    path = Path('./data/crazyhorse')

    img_path_list = sorted([str(x) for x in path.iterdir()])

    sfm_pipeline = SFM(K, img_path_list)
