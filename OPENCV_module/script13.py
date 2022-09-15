### Import OpenCV & numpy #####

import cv2

import numpy as np

import matplotlib.pyplot as plt


### Read an Image #####

img = cv2.imread('blue.jpg')


#####  Equalize  #####################

H,S,V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

eq_V = cv2.equalizeHist(V)

eq_image = cv2.merge([H,S,eq_V])

eq_image = cv2.cvtColor(eq_image, cv2.COLOR_HSV2BGR)


resized1 = cv2.resize(eq_image,(600,250),interpolation=cv2.INTER_AREA)





########################
cv2.imshow('main', resized1)
cv2.waitKey(0)



##Exiting Function######

cv2.destroyAllWindows()