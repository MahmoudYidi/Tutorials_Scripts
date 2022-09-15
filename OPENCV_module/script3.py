### Import OpenCV & numpy #####

import cv2

import numpy as np


#######################

#path = r"/"


### Read an Image #####

img1 = cv2.imread('robot1.jpg')
img2 = cv2.imread('robot2.jpg')

######################


#### Add Images #####



##### Subtract Images #####



#########################



### Display the results ###

cv2.imshow('Image 1', img1)
cv2.waitKey(0)

cv2.imshow('Image 2', img2)
cv2.waitKey(0)

cv2.imshow('Added', )
cv2.waitKey(0)

cv2.imshow('Subtract', )
cv2.waitKey(0)

########################




##Exiting Function######

cv2.destroyAllWindows()

########################