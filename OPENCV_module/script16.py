### Import OpenCV & numpy #####

import cv2

import numpy as np

import matplotlib.pyplot as plt


### Read an Image #####

img = cv2.imread('coins.jpg')

img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



#####  CIRCLE DETECTION #####################
blurred = cv2.blur(img1, (3,3))

circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 150, param1=50, param2=40, minRadius=50, maxRadius=120)

if circles is not None:
    circles = np.uint16(np.around(circles))

    for pt in circles[0,:]:
        a,b,r = pt[0], pt[1], pt[2]
        cv2.circle(img, (a,b), r, (0, 255, 0), 2)
        cv2.circle(img, (a,b), 1, (0, 0, 255), 2)


########################
cv2.imshow('main', img)
cv2.waitKey(0)




##Exiting Function######

cv2.destroyAllWindows