### Import OpenCV #####

import cv2

#######################

#path = r"/"


### Read an Image #####

img  = cv2.imread('cat.jpeg')

######################



### Display the image ###

cv2.imshow('image of a cat',img)

########################


##### Saving Image ########

filename = 'abc.png'
cv2.imwrite(filename, img)






##Exiting Function######
cv2.waitKey(0)
cv2.destroyAllWindows
########################