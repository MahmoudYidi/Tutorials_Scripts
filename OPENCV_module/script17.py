### Import OpenCV & numpy #####

import cv2

import numpy as np

import matplotlib.pyplot as plt


### Read an Image #####

img = cv2.imread('sudoku.PNG')

img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



##### Harris Corner #####################
img1 = np.float32(img1)

corner = cv2.cornerHarris(img1, 2, 5, 0.07)

corner = cv2.dilate(corner, None)

img[corner > 0.01 * corner.max()] = [0,0,255]


########################
cv2.imshow('main', img)
cv2.waitKey(0)


##Exiting Function######

cv2.destroyAllWindows