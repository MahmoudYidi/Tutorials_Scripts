### Import OpenCV & numpy #####

import cv2

import numpy as np


#######################

#path = r"/"


### Read an Image #####

img = cv2.imread('robot2.jpg')



######### Gaussian Blur ############
kernel_size = (7,7)

Gas = cv2.GaussianBlur(img,kernel_size,0)





########## Median Blur###############

kernerl_size = 7

Med = cv2.medianBlur(img,kernel_size)

### Display the results ###

cv2.imshow('Main', img)
cv2.waitKey(0)

cv2.imshow('Gaussian', )
cv2.waitKey(0)

cv2.imshow('Median', )
cv2.waitKey(0)

########################




##Exiting Function######

cv2.destroyAllWindows()

########################