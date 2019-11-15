#!/usr/bin/python2

import numpy as np
import cv2 as cv
import math
from matplotlib import pyplot as plt
from pathlib2 import Path
from itertools import compress

MIN_MATCH_COUNT = 10

path = Path('./boruszyn/images')

img_path_list = sorted([str(x) for x in path.iterdir()])

# Initiate SIFT detector

print("Obtaining keypoints and descriptors.")

def kNNMatch(img1, img2, thresh=0.9):
    kp1, des1 = img1['kp'], img1['des']
    kp2, des2 = img2['kp'], img2['des']

    if not (des1 is None) and not (des2 is None):
        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING2)#, crossCheck=True)
        # Match descriptors.
        matches = bf.knnMatch(des1, des2, k=2)

        # ratio test, stackoverflow
        good = []
        for j,(m,n) in enumerate(matches):
            if m.distance < thresh*n.distance:
                good.append(m)
        
        return good

    return []

# Initiate ORB detector
orb = cv.ORB_create()

img_info = []

for img_path in img_path_list:

    print(img_path)

    img = cv.imread(img_path, 1)

    # find the keypoints and descriptors with ORB
    kp, des = orb.detectAndCompute(cv.cvtColor(img, cv.COLOR_RGB2GRAY),None)

    img_info.append({
        "img": img,
        "kp": kp,
        "des": des,
        "M": np.eye(3)
    })

print("Obtaining distortion matrices")

match_count = np.zeros((len(img_info), len(img_info)))

for i in range(0, len(img_info)):
    for j in range(i+1, len(img_info)):
        match_count[j, i] = len(kNNMatch(img_info[i], img_info[j]))

match_count += match_count.T

match_sum = np.sum(match_count, axis=0)
order = np.flip(np.argsort(match_sum))
order = np.flip(np.argsort(match_count[order[0],:]))
order = np.hstack((order[-1],order[0:-1]))
ordered_match_list = np.flip(np.sort(match_count[order[0],:]))
print(ordered_match_list)
print(order)


base_img = img_info[order[0]]
des_list = []

# Remove the outliers (estimative)
len_order = int(math.floor(0.6*len(order)))

for i in range(1, len_order):
    print("Image "+str(i)+" of "+str(len_order-1))
    j = order[i]
    img = img_info[j]
    
    train_points = []
    query_points = []
    match_list = kNNMatch(img, base_img)
    if len(match_list)>=MIN_MATCH_COUNT:
        dst_pts = np.float32([ base_img["kp"][m.trainIdx].pt for m in match_list ]).reshape(-1,1,2)
        src_pts = np.float32([ img["kp"][m.queryIdx].pt for m in match_list ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        img_info[j]['M'] = M

    # des_list.append(img_info[order[i-1]]["des"])
    # match_list = trainFlannMatch(img, des_list)
    # if len(match_list)>=MIN_MATCH_COUNT:
    #     for m in match_list:
    #         # clouds are made of image pairs so (0,1) -> cloud_idx:0 (1,2) -> cloud_idx:1, ...
    #         if not 0 <= m.imgIdx < len_order:
    #             continue
    #         
    #         cloud_idx = m.imgIdx
    #         #if m.imgIdx != 0:
    #         #    cloud_idx = m.imgIdx-1

    #         # 2d Points
    #         # x_coords = int(img['kp'][m.queryIdx].pt[0])
    #         # y_coords = int(img['kp'][m.queryIdx].pt[1])
    #         query_points.append(img['kp'][m.queryIdx].pt)

    #         #x_coords = int(img_info[cloud_idx]['kp'][m.trainIdx].pt[0])
    #         #y_coords = int(img_info[cloud_idx]['kp'][m.trainIdx].pt[1])
    #         train_points.append(img_info[order[cloud_idx]]['kp'][m.trainIdx].pt)

    #     dst_pts = np.matrix(train_points).astype(np.float32).reshape(-1,1,2)
    #     src_pts = np.matrix(query_points).astype(np.float32).reshape(-1,1,2)
    #     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    #     # M, mask = cv.findHomography(src_pts, dst_pts, cv.LMEDS)

    #     img_info[j]['M'] = M

print("Drawing images")
M = base_img["M"]

img2 = img_info[0]["img"]

i = 2
cv.imwrite("img"+str(i-1)+".png", img2)

#x = 0
#y = 0

x_1 = 0
y_1 = 0

for i in range(0, len_order):
    j = order[i]
    image = img_info[j]

    print("Image "+str(i+1)+" of "+str(len_order))
    i += 1

    img1 = image["img"]
    M = image["M"]

    h,w,d = img1.shape
    pts = np.float32([ [0,0,1],[w,0,1],[w,h,1],[0,h,1] ]).T
    dst = M.dot(pts)
    dst = np.vstack((dst[:,0]/dst[2,0],dst[:,1]/dst[2,1],dst[:,2]/dst[2,2],dst[:,3]/dst[2,3])).T

    x_i = int(round(min(dst[0,:])))# + x# + w2 - w
    x_f = int(round(max(dst[0,:])))# + x# + w2 - w
    y_i = int(round(min(dst[1,:])))# + y# + h2 - h
    y_f = int(round(max(dst[1,:])))# + y# + h2 - h

    print(x_i, x_f)

    M[0,2] -= x_i
    M[1,2] -= y_i

    w2 = x_f - x_i
    h2 = y_f - y_i
    print((w2, h2))

    t_img1 = cv.warpPerspective(img1, M, (w2,h2), flags=cv.INTER_LINEAR)
    cv.imwrite("img"+str(i-1)+".png", t_img1)

    #pts = np.float32([ [0,0,1],[w,0,1],[w,h,1],[0,h,1] ]).T
    #dst = M2.dot(pts)
    #dst = np.vstack((dst[:,0]/dst[2,0],dst[:,1]/dst[2,1],dst[:,2]/dst[2,2],dst[:,3]/dst[2,3])).T

    #x_i = int(round(min(dst[0,:])))# + x# + w2 - w
    #y_i = int(round(min(dst[1,:])))# + y# + h2 - h

    x_i += x_1
    y_i += y_1
    x_f = x_i + w2
    y_f = y_i + h2

    # dst_2 = cv.perspectiveTransform(pts,M) #test
    # img2 = cv.polylines(img2,[np.int32(dst_2)],True,255,3, cv.LINE_AA) #test

    h3, w3, d = img2.shape

    w_f = max(x_f, w3) - min(x_i, 0)
    h_f = max(y_f, h3) - min(y_i, 0)

    img2_temp = np.zeros((h_f, w_f, d), np.uint8)
    Y_2 = max(0, -y_i)
    X_2 = max(0, -x_i)
    img2_temp[Y_2:(Y_2+h3), X_2:(X_2+w3), :] = img2

    img1_temp = np.zeros((h_f, w_f, d), np.uint8)
    Y_1 = max(0, y_i)
    X_1 = max(0, x_i)
    img1_temp[Y_1:(Y_1+h2), X_1:(X_1+w2), :] = t_img1

    x_1 += X_2
    y_1 += Y_2

    img1gray = cv.cvtColor(img1_temp, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img1gray, 1, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)

    img1_fg = cv.bitwise_and(img1_temp, img1_temp, mask = mask)
    img2_bg = cv.bitwise_and(img2_temp, img2_temp, mask = mask_inv)

    img2 = cv.add(img2_bg, img1_fg)
    #img2 = cv.polylines(img2,[np.int32(dst_2)],True,255,3, cv.LINE_AA) #test

    cv.imwrite('result.png', img2)

# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = 2)

# img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
