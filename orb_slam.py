#!/usr/bin/python2

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pathlib2 import Path

MIN_MATCH_COUNT = 10

path = Path('./Sample Banana Plant/Images')

img_path_list = sorted([str(x) for x in path.iterdir()])

img2 = cv.imread(img_path_list[0], 1) # trainImage

print(img_path_list[0])

for img_path in img_path_list[1:]:

    img1 = cv.imread(img_path, 1) # queryImage

    print(img_path)

    # Initiate ORB detector
    orb = cv.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(cv.cvtColor(img1, cv.COLOR_RGB2GRAY),None)
    kp2, des2 = orb.detectAndCompute(cv.cvtColor(img2, cv.COLOR_RGB2GRAY),None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING2)#, crossCheck=True)
    # Match descriptors.
    matches = bf.knnMatch(des2, des1, k=2)
    # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)

    # ratio test, stackoverflow
    good = []
    for j,(m,n) in enumerate(matches):
        if m.distance < 0.90*n.distance:
            good.append(m)

    if len(good)>=MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w,d = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        w2 = int(max(dst[:,:,0])-min(dst[:,:,0]))
        h2 = int(max(dst[:,:,1])-min(dst[:,:,1]))

        x_i = min(dst[:,:,0])# + w2 - w
        x_f = max(dst[:,:,0])# + w2 - w
        y_i = min(dst[:,:,1])# + h2 - h
        y_f = max(dst[:,:,1])# + h2 - h

        M[0,2] -= x_i
        M[1,2] -= y_i

        t_img1 = cv.warpPerspective(img1, M, (w2,h2))

        # cv.imwrite('test.png', t_img1)

        # dst_2 = cv.perspectiveTransform(pts,M) #test
        # img2 = cv.polylines(img2,[np.int32(dst_2)],True,255,3, cv.LINE_AA) #test

        h3, w3, d = img2.shape

        w_f = int(max(x_f, w3) - min(x_i, 0))
        h_f = int(max(y_f, h3) - min(y_i, 0))

        img2_temp = np.zeros((h_f, w_f, d), np.uint8)
        img2_temp[max(0, -y_i):int(max(0, -y_i)+h3), max(0, -x_i):int(max(0, -x_i)+w3), :] = img2

        img1_temp = np.zeros((h_f, w_f, d), np.uint8)
        img1_temp[max(0, y_i):(max(0, y_i)+h2), max(0, x_i):(max(0, x_i)+w2), :] = t_img1

        img1gray = cv.cvtColor(img1_temp, cv.COLOR_BGR2GRAY)
        ret, mask = cv.threshold(img1gray, 1, 255, cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)

        img1_fg = cv.bitwise_and(img1_temp, img1_temp, mask = mask)
        img2_bg = cv.bitwise_and(img2_temp, img2_temp, mask = mask_inv)

        img2 = cv.add(img2_bg, img1_fg)
        #img2 = cv.polylines(img2,[np.int32(dst_2)],True,255,3, cv.LINE_AA) #test
        cv.imwrite('result.png', img2)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    # draw_params = dict(matchColor = (0,255,0),
    #                    singlePointColor = (255,0,0),
    #                    matchesMask = matchesMask,
    #                    flags = 2)

    # img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
