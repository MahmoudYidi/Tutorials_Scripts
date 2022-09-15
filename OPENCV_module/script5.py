### Import OpenCV & numpy #####

import cv2

import numpy as np


#######################

#path = r"/"


### Read an Image #####

img = cv2.imread('robot2.jpg')

######################

#### Create a Kernel #####

kernel = np.ones((6,6),np.uint8)



###### Erode Image ##### 
eroded = cv2.erode(img, kernel)

###### Dilate ########
Dilate = cv2.dilate(img,kernel)




### Display the results ###

cv2.imshow('Main', img)
cv2.waitKey(0)

cv2.imshow('Erroded',eroded)
cv2.waitKey(0)



cv2.imshow('Dilate', Dilate )
cv2.waitKey(0)

########################


##Exiting Function######

cv2.destroyAllWindows()

########################