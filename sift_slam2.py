#!/usr/bin/python2

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pathlib2 import Path

MIN_MATCH_COUNT = 10

path = Path('./Sample Banana Plant/Images')

img_path_list = sorted([str(x) for x in path.iterdir()])

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()

img_info = []

print("Obtaining keypoints and descriptors.")

for img_path in img_path_list:

    print(img_path)

    img = cv.imread(img_path, 1)

    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(cv.cvtColor(img, cv.COLOR_RGB2GRAY),None)

    img_info.append({
        "img": img,
        "kp": kp,
        "des": des,
        "M": None,
        "mask": None
    })

print("Obtaining distortion matrices")

img_info[0]["M"] = np.eye(3)

match_list = []

for i in range(1, len(img_info)):

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(img_info[i-1]["des"], img_info[i]["des"], k=2)
    #matches = flann.knnMatch(img_info[i-1]["des"], img_info[i]["des"], k=2)
    good = []
    # ratio test as per Lowe's paper
    for j,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>=MIN_MATCH_COUNT:
        match_list.append(good)
        dst_pts = np.float32([ img_info[i]["kp"][m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        src_pts = np.float32([ img_info[i-1]["kp"][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        # M, mask = cv.findHomography(src_pts, dst_pts, cv.LMEDS)
        img_info[i]["M"] = M
        img_info[i]["mask"] = mask
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        img_info[i]["M"] = np.eye(3)


M = img_info[0]["M"]

img2 = img_info[0]["img"]

i = 2
cv.imwrite("img"+str(i-1)+".png", img2)

#x = 0
#y = 0

max_i = len(img_info)

for image in img_info[1:]:
    print("Image "+str(i)+" of "+str(max_i))
    i += 1

    img1 = image["img"]
    #M = image["M"] 
    #M = M.dot(image["M"])
    M2 = M.dot(np.linalg.inv(image["M"]))
    M2 = M2/M2[2,2]

    M = M.dot(image["M"])
    M = M/M[2,2]
    #mask = image["mask"]

    h,w,d = img1.shape
    pts = np.float32([ [0,0,1],[w,0,1],[w,h,1],[0,h,1] ]).T
    dst = M.dot(pts)
    dst = np.vstack((dst[:,0]/dst[2,0],dst[:,1]/dst[2,1],dst[:,2]/dst[2,2],dst[:,3]/dst[2,3])).T
    #dst = cv.perspectiveTransform(pts,M)

    #x_i = int(min(dst[:,:,0]))# + x# + w2 - w
    #x_f = int(max(dst[:,:,0]))# + x# + w2 - w
    #y_i = int(min(dst[:,:,1]))# + y# + h2 - h
    #y_f = int(max(dst[:,:,1]))# + y# + h2 - h
    x_i = int(round(min(dst[0,:])))# + x# + w2 - w
    x_f = int(round(max(dst[0,:])))# + x# + w2 - w
    y_i = int(round(min(dst[1,:])))# + y# + h2 - h
    y_f = int(round(max(dst[1,:])))# + y# + h2 - h

    print(x_i, x_f)

    M_orig = M

    M[0,2] -= x_i
    M[1,2] -= y_i

    w2 = x_f - x_i
    h2 = y_f - y_i
    print((w2, h2))

    t_img1 = cv.warpPerspective(img1, M, (w2,h2), flags=cv.INTER_LINEAR)
    cv.imwrite("img"+str(i-1)+".png", t_img1)

    M = M_orig

    pts = np.float32([ [0,0,1],[w,0,1],[w,h,1],[0,h,1] ]).T
    dst = M2.dot(pts)
    dst = np.vstack((dst[:,0]/dst[2,0],dst[:,1]/dst[2,1],dst[:,2]/dst[2,2],dst[:,3]/dst[2,3])).T

    x_i = int(round(min(dst[0,:])))# + x# + w2 - w
    x_f = x_i + w2
    y_i = int(round(min(dst[1,:])))# + y# + h2 - h
    y_f = y_i + h2
    
    #x_i += x_i
    #x_f += x_i 
    #y_i += y_i
    #y_f += y_i
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
