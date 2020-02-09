#!/usr/bin/python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pathlib2 import Path
import os
import sys
from bundle_adj2 import adjust

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')

class SFM(object):
    def __init__(self, instrinsic, images_path, distCoeffs = 0):
        self.distCoeffs = distCoeffs
        self.MIN_MATCH_COUNT=10
        self.K = instrinsic

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

            camera_pose = self.estimate_new_view_pose(img)
            points2, points1, matches = self.kNNMatch(img, self.img_data[prev_img_idx])
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

            points_3d = self.triangulatePoints(self.img_data[prev_img_idx]['pose'], camera_pose,
                                               points1, points2)

            any_nan = np.array(np.any(np.isnan(points_3d), axis=-1))
            all_nan = np.array(np.all(np.isnan(points_3d), axis=-1))
            print(str(self.imgs_used)+": points3d - number of NaNs:"+str(len(any_nan[any_nan])))
            print(str(self.imgs_used)+": points3d - number of NaNs:"+str(len(all_nan[all_nan])))

            points_idx = [x.queryIdx for x in matches]
            self.img_data[self.imgs_used]['pose'] = camera_pose

            points_2d, colors = self.get_2dpoints_and_colors_from_img(self.img_data[self.imgs_used], points2)
            self.imgs_used += 1

            point_cloud_data = {'3dpoints': points_3d,
                                '2dpoints': points_2d,
                                'point_img_corresp': points_idx,
                                'colors': colors}

            self.point_cloud.append(point_cloud_data)

        # Bundle Adjustment
        print("Adjusting...")
        [_, dx_p, point_list, color_list] = adjust(self.point_cloud, K, [data['pose'] for data in self.img_data], self.imgs_used)
        self.write_ply(point_list, color_list, "no_ba.ply")
        point_list = point_list + dx_p

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

    def trainFlannMatch(self, img, current_descriptors, lowes_thresh=0.7):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

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

    def kNNMatch(self, img1, img2, lowes_thresh=0.7):
        kp1, des1 = img1['keypoints'], img1['descriptors']
        kp2, des2 = img2['keypoints'], img2['descriptors']

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

        search_params = dict(checks=50)   # or pass empty dictionary
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

            if len(good)>=self.MIN_MATCH_COUNT:
                points1 = np.array([kp1[x.queryIdx].pt for x in good])
                points2 = np.array([kp2[x.trainIdx].pt for x in good])
                return points1, points2, good

        return [], [], []

    def findDecomposedEssentialMatrix(self, p1, p2):
        # fundamental matrix and inliers
        # F, mask = cv.findFundamentalMat(p1, p2, cv.FM_LMEDS, 1, 0.999)
        F, mask = cv.findFundamentalMat(p1, p2, cv.FM_RANSAC, 1.0, 0.999)
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
        
        # points_4d_hom = cv.triangulatePoints(P1, P2, points1.T, points2.T)
        # points_4d_hom = cv.triangulatePoints(P2, P1, pts2_norm, pts1_norm)
        # pts1_norm = points1.T
        # pts2_norm = points2.T
        # points_4d_hom = cv.triangulatePoints(self.K.dot(P1), self.K.dot(P2), pts1_norm, pts2_norm)
        points_3d = cv.convertPointsFromHomogeneous(points_4d_hom.T).reshape(-1,3)
        # points_4d = points_4d_hom / np.tile(points_4d_hom[-1, :], (4, 1))
        # points_3d = points_4d[:3, :].T
        return points_3d

    def compute_initial_cloud(self, img1, img2):
        ''' Keypoint Matching '''
        points2, points1, matches = self.kNNMatch(img2, img1)

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

        return [point_cloud_data1, point_cloud_data2]

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

        # P1 = self.img_data[self.imgs_used-1]["pose"]
        # P1_ext = np.vstack((P1, (1, 1, 1, 1)))
        # P_tmp = np.hstack((R, t))
        # P_tmp = np.vstack((P_tmp, (1, 1, 1, 1)))
        # # P2 = P_tmp.dot(P1_ext)
        # P2 = P1_ext.dot(P_tmp)
        # for i in range(4):
        #     P2[:,i] /= P2[3,i]
        # P2 = P2[:3, :]

        P2 = np.hstack((R2, t2))

        # P1 = self.img_data[self.imgs_used-1]["pose"]
        # #P2 = np.hstack((np.dot(R, P1[:,:3]), (t.T + P1[:,3]).T))
        # R1 = P1[:,:3]
        # t1 = P1[:,3]
        # # P2 = np.hstack((np.dot(R, R1), (t.T + R.dot(t1)).T))
        # P2 = np.hstack((np.dot(R1, R), (R1.dot(t).T + t1).T))

        print(P2)

        return P2, points1, points2, matches

    def estimate_new_view_pose(self, img):
        # descriptors = [uImg['descriptors']
        #         for uImg in self.img_data[:self.imgs_used]]
        
        descriptors = [self.img_data[self.imgs_used-1]['descriptors']]

        # for uImg in self.img_data[:self.imgs_used]:
        #     descriptors.append(uImg['descriptors'])

        prev_pose = self.img_data[self.imgs_used-1]['pose']

        matches = self.trainFlannMatch(img, descriptors)
        # matches = self.kNNMatch(self.img_data[self.imgs_used-1], img)

        # 3d Points
        points_3d = []
        points_2d = []
        for m in matches:
            # cloud_idx = m.imgIdx
            cloud_idx = self.imgs_used - 1

            # search function
            pointIdx = np.asarray(np.asarray(self.point_cloud[cloud_idx]['point_img_corresp']) == m.trainIdx).nonzero()
            if len(pointIdx[0]) == 0:
                continue

            # Get the 3d Point corresponding to the train image keypoint
            points_3d.append(self.point_cloud[cloud_idx]['3dpoints'][pointIdx][0])

            # 2d Points
            x_coords = int(img['keypoints'][m.queryIdx].pt[0])
            y_coords = int(img['keypoints'][m.queryIdx].pt[1])
            points_2d.append([x_coords, y_coords])
            if len(points_3d) == 30:
                break

        #camera_pose = np.zeros((3,4))
        # estimate camera pose from 3d2d Correspondences
        if len(points_3d) > 4 and len(points_2d) > 4:
            rvec_in, _ = cv.Rodrigues(prev_pose[:,:3])
            tvec_in = prev_pose[:,3]
            # _, rvec, tvec = cv.solvePnP(
            #                      np.array(points_3d, dtype=np.float64),
            #                      np.array(points_2d, dtype=np.float64),
            #                      self.K, self.distCoeffs, flags=cv.SOLVEPNP_ITERATIVE)
            #                      #useExtrinsicGuess=True, rvec=rvec_in, tvec=tvec_in)
            _, rvec, tvec, inliers = cv.solvePnPRansac(
                                np.array(points_3d, dtype=np.float64),
                                np.array(points_2d, dtype=np.float64),
                                self.K, self.distCoeffs, confidence=0.999999,
                                flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=10000,
                                useExtrinsicGuess=True, rvec=rvec_in, tvec=prev_pose[:,3],
                                reprojectionError=5.0)
        

            # RANSAC        
            # ITERATIVE     
            # P3P           
            # EPNP          
            # DLS           
            # UPNP          
            # AP3P          
            # MAX_COUNT     

            cam_rmat, _ = cv.Rodrigues(rvec)
            camera_pose = np.concatenate([cam_rmat.get(), tvec.get()], axis=1)
            # R1 = prev_pose[:, :3]
            # t1 = prev_pose[:, 3]
            # R2 = np.dot(cam_rmat, R1)
            # t2 = (np.dot(cam_rmat, t1) + tvec.T).T
            # camera_pose = np.hstack((R2, t2))
        else:
            camera_pose = np.zeros((3,4))
        #    camera_pose[:,:3] = cam_rmat
        #    camera_pose[:,3] = tvec.T
        return camera_pose


    def get_2dpoints_and_colors_from_img(self, img, matches):
        # x_coords = [int(img['keypoints'][x.queryIdx].pt[0]) for x in matches]
        # y_coords = [int(img['keypoints'][x.queryIdx].pt[1]) for x in matches]
        x_coords = [int(x[0]) for x in matches]
        y_coords = [int(x[1]) for x in matches]
        image_coords = np.column_stack([x_coords, y_coords])
        colors = img['pixels'][y_coords, x_coords, :]
        return image_coords, colors


    def write_ply(self, points, colors, name='mesh.ply'):
        #ply_header = (
        #            '''ply
        #            format ascii 1.0
        #            element vertex {vertex_count}
        #            property float x
        #            property float y
        #            property float z
        #            property uchar red
        #            property uchar green
        #            property uchar blue
        #            end_header
        #            '''
        #            )
        filename = 'meshes/'+name

        points = np.hstack([points, colors])
        with open(filename, 'w') as outfile:
            #outfile.write(ply_header.format(vertex_count=len(coords)))
            np.savetxt(outfile, points, '%f %f %f %d %d %d')

if __name__  == '__main__':

    if not os.path.isdir("./meshes"):
        os.mkdir("meshes")

    # K = np.array([[2759.48, 0,       1520.69],
    #               [0,       2764.16, 1006.81],
    #               [0,       0,       1]])
    #
    # path = Path('./data/fountain-P11/images')

    path = Path('./data/crazyhorse')
    distCoeffs = 0
    f = 2500.0
    width = 1024.0
    height = 768.0
    K = np.array([[f,0,width/2],
                  [0,f,height/2],
                  [0,0,1]])

    # path = Path('./images')

    # K = np.array([[3140.63, 0, 1631.5],
    #               [0, 3140.63, 1223.5],
    #               [0, 0, 1]])

    # GoPro
    # path = Path('./images-2')
    # camera = np.load('./calibration_data.npz')
    # K = camera['intrinsic_matrix']
    # distCoeffs = camera['distCoeff']

    # Castle
    # path = Path('./castle')
    # K = np.array([[2905.88, 0, 1416],
    #               [0, 2905.88, 1064],
    #               [0, 0, 1]])
    # distCoeffs = 0

    # Celular
    # path = Path('./images-lantern2')
    # camera = np.load('./camera.npz')
    # K = camera['mtx']
    # distCoeffs = camera['dist']

    img_path_list = sorted([str(x) for x in path.iterdir()])

    sfm_pipeline = SFM(K, img_path_list, distCoeffs=distCoeffs)
