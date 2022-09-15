### Import OpenCV  #####

import cv2

#######################




### Read an Image #####

image = cv2.imread('robot2.jpg')


##### Dimension of Image ########

dimension = image.shape

height = image.shape[0]

width = image.shape[1]



###### Scaling ##########
scale = 50/100

height = int(height*scale)

width = int(scale*width)

New_dim = (width,height)

##### Resize ###########

resized = cv2.resize(image, New_dim, interpolation=cv2.INTER_AREA)
### Display the image ###

cv2.imshow('Main Image', image)
cv2.waitKey(0)

cv2.imshow('resized', resized)
cv2.waitKey(0)

########################



##Exiting Function######

cv2.destroyAllWindows()

########################