### Import OpenCV #####

import cv2

#######################

#path = r"/"


### Read an Image #####
img = cv2.imread('rgb.png')

######################


#### Split images into 3 colour spaces BGR ####

B, G, R =cv2.split(img)



### Display the image ###
cv2.imshow('Main',img)
cv2.waitKey(0)
cv2.imshow('Blue',B)
cv2.waitKey(0)
cv2.imshow('Green',G)
cv2.waitKey(0)
cv2.imshow('Red',R)
cv2.waitKey(0)


########################



##Exiting Function######

cv2.destroyAllWindows()

########################