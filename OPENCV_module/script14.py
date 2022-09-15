### Import OpenCV & numpy #####

import cv2

import numpy as np

import matplotlib.pyplot as plt


### Read an Image #####

img = cv2.imread('mix.jpg')


#####  FILTER AN OBJECT BASED ON COLOUR #####################
H = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([90,9,248])
upper = np.array([130,68, 255])

mask = cv2.inRange(H, lower, upper)

result = cv2.bitwise_and(img, img, mask = mask)


########################
cv2.imshow('main', img)
cv2.waitKey(0)

cv2.imshow('mask', mask)
cv2.waitKey(0)

cv2.imshow('result', result)
cv2.waitKey(0)




##Exiting Function######

cv2.destroyAllWindows()