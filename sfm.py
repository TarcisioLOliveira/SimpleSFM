#!/usr/bin/python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pathlib2 import Path
import os
import sys
from bundle_adj2 import adjust
import scipy.sparse
import scipy.optimize
from scipy.sparse import linalg
from scipy.linalg import interpolative
from scipy.optimize import Bounds

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')

REPROJ_ERROR = 10
RANSAC_MAXITER = 5000
RANSAC_CONFIDENCE = 0.995
PNP_POINTS = 1000000000

class SFM(object):
    def __init__(self, instrinsic, images_path, distCoeffs = 0):
        global REPROJ_ERROR
        self.distCoeffs = distCoeffs
        self.MIN_MATCH_COUNT=10
        self.K = instrinsic

        self.base_mean = 0

        img_path_list = sorted([str(x) for x in path.iterdir()])
        self.img_data = self.read_and_compute_keypoints(img_path_list)
        self.point_cloud = self.compute_initial_cloud(self.img_data[0], self.img_data[1])
        self.imgs_used = 2
        n_cameras = 1

        max_poses = len(img_path_list) - 1
        print('Initial conditions established')
        for img in self.img_data[2:]:
            print('New pose estimation, '+str(self.imgs_used)+' of '+str(max_poses))
            # camera_pose, points1, points2, matches = self.old_estimate_new_view_pose(img)
            prev_img_idx = self.imgs_used - 1

            camera_pose, points1, points2, matches  = self.estimate_new_view_pose(img)#, points1, points2, points_idx = self.estimate_new_view_pose(img)
            if len(points1) == 0 or len(points2) == 0 or not np.any(camera_pose):
                print("Not enough matches: "+str(len(matches))+"/"+str(self.MIN_MATCH_COUNT))
                self.img_data[self.imgs_used]['pose'] = camera_pose
                self.imgs_used += 1
                point_cloud_data = {'3dpoints': [],
                                    '2dpoints': [],
                                    'point_img_corresp': [],
                                    'colors': []}
                self.point_cloud.append(point_cloud_data)
                continue


            _, mask = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC, REPROJ_ERROR, RANSAC_CONFIDENCE, RANSAC_MAXITER)
            mask = mask.astype(bool).flatten()
            points1 = points1[mask]
            points2 = points2[mask]
            matches = np.asarray(matches)[mask]
            points_idx = [x.queryIdx for x in matches]

            P2, points_3d = self.triangulatePoints(self.img_data[prev_img_idx]['pose'], camera_pose,
                                               points1, points2)

            #points_3d, points2, points_idx = self.remove_farther_points(camera_pose, points_3d, points2, points_idx)

            #points_3d, points_idx, points2 = self.check_triangulation(points_3d, points1, points2, self.img_data[prev_img_idx]['pose'], camera_pose, matches, points2)

            any_nan = np.array(np.any(np.isnan(points_3d), axis=-1))
            all_nan = np.array(np.all(np.isnan(points_3d), axis=-1))
            print(str(self.imgs_used)+": points3d - number of NaNs:"+str(len(any_nan[any_nan])))
            print(str(self.imgs_used)+": points3d - number of NaNs:"+str(len(all_nan[all_nan])))

            self.img_data[self.imgs_used]['pose'] = camera_pose

            points_2d, colors = self.get_2dpoints_and_colors_from_img(self.img_data[self.imgs_used], points2)
            self.imgs_used += 1

            point_cloud_data = {'3dpoints': points_3d,
                                '2dpoints': points_2d,
                                'point_img_corresp': points_idx,
                                'colors': colors}

            self.point_cloud.append(self.prune_points(point_cloud_data))

        # Bundle Adjustment
        #print("Adjusting...")
        point_list = self.point_cloud[0]['3dpoints']
        color_list = self.point_cloud[0]['colors']
        for i in range(1, len(self.point_cloud)):
            if(len(self.point_cloud[i]['3dpoints']) > 0):
                point_list = np.vstack((point_list, self.point_cloud[i]['3dpoints']))
                color_list = np.vstack((color_list, self.point_cloud[i]['colors']))

        print(point_list.shape)
        print(color_list.shape)

        self.write_ply(point_list, color_list)

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

    def trainFlannMatch(self, img, current_descriptors, lowes_thresh=0.8):
        FLANN_INDEX_KDTREE = 1
        FLANN_INDEX_KMEANS = 2
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm = FLANN_INDEX_KMEANS, trees = 20)
        search_params = dict(checks=100)   # or pass empty dictionary

        flann = cv.FlannBasedMatcher(index_params, search_params)

        # each element is a set of descriptors from an image
        flann.add(current_descriptors)
        flann.train()
        
        size = 0
        for d in current_descriptors:
            size += len(d)

        # for each descriptor in the query, find the closest match
        matches = flann.match(queryDescriptors=img['descriptors'])
        matches = sorted(matches, key= lambda x:x.distance)
        return matches#[:int(0.01*size)]

    def kNNMatch(self, img1, img2, lowes_thresh=0.50):
        kp1, des1 = img1['keypoints'], img1['descriptors']
        kp2, des2 = img2['keypoints'], img2['descriptors']

        FLANN_INDEX_KDTREE = 1
        FLANN_INDEX_KMEANS = 2
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm = FLANN_INDEX_KMEANS, trees = 20)

        search_params = dict(checks=100)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        if not (des1 is None) and not (des2 is None):
            matches = flann.knnMatch(des1,des2,k=2)

            # Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for j in range(len(matches))]

            # ratio test as per Lowe's paper
            good = []
            for j,(m,n) in enumerate(matches):
                if m.distance < lowes_thresh*n.distance:
                    good.append(m)

            good = sorted(good, key= lambda x:x.distance)
            if len(good)>=self.MIN_MATCH_COUNT:
                points1 = np.array([kp1[x.queryIdx].pt for x in good])
                points2 = np.array([kp2[x.trainIdx].pt for x in good])
                return points1, points2, good

        return [], [], []

    def findDecomposedEssentialMatrix(self, p1, p2):
        # fundamental matrix and inliers
        # F, mask = cv.findFundamentalMat(p1, p2, cv.FM_LMEDS, 1, 0.999)
        F, mask = cv.findFundamentalMat(p1, p2, cv.FM_RANSAC, REPROJ_ERROR, RANSAC_CONFIDENCE, RANSAC_MAXITER)
        mask = mask.astype(bool).flatten()
        E = np.dot(self.K.T, np.dot(F, self.K))

        _, R, t, _ = cv.recoverPose(E, p1[mask], p2[mask], self.K)

        return R, t, p1[mask], p2[mask], mask

    def triangulatePoints(self, P1, P2, points1, points2):

        pts1_norm = cv.undistortPoints(np.expand_dims(points1, axis=1),
                cameraMatrix=self.K, distCoeffs=self.distCoeffs)
        pts2_norm = cv.undistortPoints(np.expand_dims(points2, axis=1),
                cameraMatrix=self.K, distCoeffs=self.distCoeffs)

        points_4d_hom = cv.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
        points_3d = cv.convertPointsFromHomogeneous(points_4d_hom.T).reshape(-1,3)

        return P2, points_3d

    def compute_initial_cloud(self, img1, img2):
        ''' Keypoint Matching '''
        points2, points1, matches = self.kNNMatch(img2, img1)

        if len(points1) == 0:
            print("Not enough matches: "+str(len(matches))+"/"+str(self.MIN_MATCH_COUNT))
            return None

        ''' Param Estimation '''
        R, t, points1, points2, mask = self.findDecomposedEssentialMatrix(points1, points2)

        P1 = np.column_stack([np.eye(3), np.zeros(3)])
        P2 = np.hstack((R, t))

        ''' Triangulation '''
        P2, points_3d = self.triangulatePoints(P1, P2, points1, points2)
        # ids of the matches used
        matches = np.asarray(matches)[mask]
        points_idx1 = [x.trainIdx for x in matches]
        points_idx2 = [x.queryIdx for x in matches]

        self.img_data[0]['pose'] = P1
        self.img_data[1]['pose'] = P2

        points_2d1, colors1 = self.get_2dpoints_and_colors_from_img(img1, points1)
        points_2d2, colors2 = self.get_2dpoints_and_colors_from_img(img2, points2)

        point_cloud_data1 = {'3dpoints': points_3d,
                            '2dpoints': points_2d1,
                            'point_img_corresp': points_idx1,
                            'colors': colors1}
        point_cloud_data2 = {'3dpoints': points_3d,
                            '2dpoints': points_2d2,
                            'point_img_corresp': points_idx2,
                            'colors': colors2}

        return [self.prune_points(point_cloud_data1), self.prune_points(point_cloud_data2)]

    def old_estimate_new_view_pose(self, img):
        ''' Keypoint Matching '''
        points2, points1, matches = self.kNNMatch(img, self.img_data[self.imgs_used-1])

        if len(points1) == 0:
            print("Not enough matches: "+str(len(matches))+"/"+str(self.MIN_MATCH_COUNT))
            return None

        ''' Param Estimation '''
        P1 = self.img_data[self.imgs_used-1]["pose"]
        R1 = P1[:, :3]
        t1 = P1[:, 3]
        R, t = self.findDecomposedEssentialMatrix(points1, points2)
        R2 = np.dot(R, R1)
        t2 = (np.dot(R, t1) + t.T).T

        P2 = np.hstack((R2, t2))

        print(P2)

        return P2, points1, points2, matches

    def estimate_new_view_pose(self, img):
        prev_pose = self.img_data[self.imgs_used-1]['pose']

        points2, points1, matches = self.kNNMatch(img, self.img_data[self.imgs_used-1])

        # 3d Points
        points_3d = []
        points_2d = []
        mask = []
        for m in matches:
            # cloud_idx = m.imgIdx
            cloud_idx = self.imgs_used - 1

            # search function
            pointIdx = np.asarray(np.asarray(self.point_cloud[cloud_idx]['point_img_corresp']) == m.trainIdx).nonzero()
            if len(pointIdx[0]) == 0:
                mask.append(False)
                continue

            mask.append(True)

            # Get the 3d Point corresponding to the train image keypoint
            points_3d.append(self.point_cloud[cloud_idx]['3dpoints'][pointIdx[0][0]])

            # 2d Points
            x_coords = int(round(img['keypoints'][m.queryIdx].pt[0]))
            y_coords = int(round(img['keypoints'][m.queryIdx].pt[1]))
            points_2d.append([x_coords, y_coords])

            if len(points_3d) >= PNP_POINTS:
                break

        mask_orig_size = len(mask)
        padding = [False]*(len(matches)-len(mask))
        mask.extend(padding)
        #camera_pose = np.zeros((3,4))
        # estimate camera pose from 3d2d Correspondences
        if len(points_3d) >= 4 and len(points_2d) >= 4:
            points_3d = np.array(points_3d, dtype=np.float64)
            points_2d = np.array(points_2d, dtype=np.float64)

            rvec_in, _ = cv.Rodrigues(prev_pose[:,:3])
            tvec_in = np.asarray(prev_pose[:,3], dtype=np.float32)

            _, rvec, tvec, inliers = cv.solvePnPRansac(
                                points_3d,
                                points_2d,
                                self.K, self.distCoeffs, confidence=RANSAC_CONFIDENCE,
                                flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=RANSAC_MAXITER,
                                #useExtrinsicGuess=True, rvec=rvec_in, tvec=tvec_in,
                                reprojectionError=REPROJ_ERROR)


            cam_rmat, _ = cv.Rodrigues(rvec)
            camera_pose = np.concatenate([cam_rmat, tvec], axis=1)

            i = 0
            j = 0
            count = 0

        else:
            camera_pose = np.zeros((3,4))

        return camera_pose, points1, points2, matches


    def get_2dpoints_and_colors_from_img(self, img, matches):
        # x_coords = [int(img['keypoints'][x.queryIdx].pt[0]) for x in matches]
        # y_coords = [int(img['keypoints'][x.queryIdx].pt[1]) for x in matches]
        x_coords = [int(x[0]) for x in matches]
        y_coords = [int(x[1]) for x in matches]
        image_coords = np.column_stack([x_coords, y_coords])
        colors = img['pixels'][y_coords, x_coords, :]
        return image_coords, colors

    def prune_points(self, point_cloud_part, max_deviation = 1.0):
        if(len(point_cloud_part['3dpoints']) == 0):
            return point_cloud_part

        z_list = np.asarray(point_cloud_part['3dpoints'].T[2])
        mean = np.mean(z_list)
        sd = np.std(z_list)
        print(mean)
        print(sd)
        max_z = mean + max_deviation*sd
        min_z = mean - max_deviation*sd
        tmp1 = np.greater_equal(z_list, min_z)
        tmp2 = np.less_equal(z_list, max_z)
        select = np.logical_and(tmp1, tmp2)
        new_3dpoints = np.asarray(point_cloud_part['3dpoints'])[select]

        point_cloud_part_pruned = {'3dpoints': new_3dpoints,
                                   '2dpoints': np.asarray(point_cloud_part['2dpoints'])[select],
                                   'point_img_corresp': np.asarray(point_cloud_part['point_img_corresp'])[select],
                                   'colors': np.asarray(point_cloud_part['colors'])[select]}

        return point_cloud_part_pruned

        #return point_cloud_part

    def remove_farther_points(self, P, points3d, points2d, points_idx):
        t = P[:, 3]
        R = P[:, :3]

        n = -np.dot(np.linalg.inv(R), t).ravel()
        mean = 0
        vectors = np.zeros((len(points3d), 3))

        for i in range(len(points3d)):
            vectors[i] = points3d[i] - n;
            mean += np.linalg.norm(vectors[i])

        mean = mean/len(points3d)
        norms = np.linalg.norm(vectors, axis=-1)
        sd = np.std(norms)
        mask = np.less_equal(norms, mean)

        return np.asarray(points3d)[mask], np.asarray(points2d)[mask], np.asarray(points_idx)[mask]

        #return point_cloud_part

    def check_triangulation(self, points3d, points1, points2, P1, P2, matches, points2d):
        rvec1, _ = cv.Rodrigues(P1[:, :3])
        tvec1 = P1[:, 3]
        rvec2, _ = cv.Rodrigues(P2[:, :3])
        tvec2 = P2[:, 3]

        reproj1, _ = cv.projectPoints(points3d, rvec1, tvec1, self.K, self.distCoeffs)
        reproj2, _ = cv.projectPoints(points3d, rvec2, tvec2, self.K, self.distCoeffs)

        reproj_error1 = points1 - reproj1.reshape(-1, 2)
        reproj_error2 = points2 - reproj2.reshape(-1, 2)

        points_idx = []
        final_cloud = []
        points_2d = []
        for i in range(len(points1)):
            if np.linalg.norm(reproj_error1[i]) + np.linalg.norm(reproj_error2[i]) < REPROJ_ERROR:
                final_cloud.append(points3d[i])
                points_idx.append(matches[i].queryIdx)
                points_2d.append(points2d[i])

        return np.array(final_cloud), points_idx, np.array(points_2d)


    def write_ply(self, points, colors, name='mesh.ply'):
        # ply_header = ("ply\nformat ascii 1.0\nelement vertex {}\n"
        #            "property float x\nproperty float y\nproperty float z\n"
        #            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        #            "end_header\n"
        #            ).format(points.shape[0])
        filename = 'meshes/'+name
        points = np.hstack([points, colors])
        with open(filename, 'a') as outfile:
            # outfile.write(ply_header)
            #outfile.write(ply_header.format(vertex_count=len(coords)))
            np.savetxt(outfile, points, '%f %f %f %d %d %d')

if __name__  == '__main__':

    if not os.path.isdir("./meshes"):
        os.mkdir("meshes")

    path = Path('./data/crazyhorse')
    distCoeffs = 0
    f = 825.26
    width = 1024.0
    height = 768.0
    K = np.array([[f,0,width/2],
                  [0,f,height/2],
                  [0,0,1]])

    #Castle
    #path = Path('./castle')

    #K = np.array([[2905.88, 0, 1416],
    #              [0, 2905.88, 1064],
    #              [0, 0, 1]])
    #distCoeffs = 0

    img_path_list = sorted([str(x) for x in path.iterdir()])

    sfm_pipeline = SFM(K, img_path_list, distCoeffs=distCoeffs)
