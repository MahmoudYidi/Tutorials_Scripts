

import cv2
import numpy as np

#Read images ########################################################################
img2 =cv2.imread('IMG_1879.jpg')
img1 = cv2.imread('IMG_1880.jpg')

#img1 =cv2.imread('IMG_1884.jpg')
#img2 = cv2.imread('IMG_1885.jpg')

#img1 =cv2.imread('result.jpg')
#img2 = cv2.imread('IMG_1886.jpg')


img1 = cv2.resize(img1,(1000,1000),interpolation=cv2.INTER_AREA)

img2 = cv2.resize(img2,(1000,1000),interpolation=cv2.INTER_AREA)

########################################################################################

descriptor = cv2.SIFT_create()
(kps1, features1) = descriptor.detectAndCompute(img1, None)
(kps2, features2) = descriptor.detectAndCompute(img2, None)

########################################################################################

points1_ = cv2.drawKeypoints(img1, kps1, outImage=None, color=(0,0,255), flags = 0)
points2_ = cv2.drawKeypoints(img2, kps2, outImage=None, color=(0,255,0), flags = 0)

########################################################################################
cv2.imwrite('point1.jpg', points1_)
cv2.imshow('Detected points image 1', points1_)
cv2.waitKey(0)
cv2.imwrite('point2.jpg', points2_)
cv2.imshow('Detected points image 2', points2_)
cv2.waitKey(0)

########################################################################################
bf = cv2.BFMatcher()
matches = bf.knnMatch(features1,features2,k=2)

feautures_match= []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        feautures_match.append(m)

MIN=5

if len(feautures_match)>MIN:
        src_pts = np.float32([ kps1[m.queryIdx].pt for m in feautures_match ]).reshape(-1,1,2)
        dst_pts = np.float32([ kps2[m.trainIdx].pt for m in feautures_match ]).reshape(-1,1,2)

        (M, mask) = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        (h,w,z) = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
    
else:
    print ('Not enough matches are found - %d/%d', (len(feautures_match),MIN))
    matchesMask = None
    
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
               singlePointColor = None,
               matchesMask = matchesMask, # draw only inliers
               flags = 2)

img3 = cv2.drawMatches(img1,kps1,img2,kps2,feautures_match,None,**draw_params)
    

cv2.imwrite('Homography.jpg', img3)
cv2.imshow('Homography', img3)
cv2.waitKey(0)

########################################################################################


results =cv2.warpPerspective(img1,M,(img1.shape[1]+img2.shape[0],img2.shape[0]))
results[0:img2.shape[0], 0:img2.shape[1]] = img2


########################################################################################
cv2.imwrite('result.jpg', results)
cv2.imshow('result', results)
cv2.waitKey(0)

cv2.destroyAllWindows()




