### Import OpenCV & numpy #####

import cv2
from cv2 import calcHist

import numpy as np

import matplotlib.pyplot as plt


#######################

#path = r"/"


### Read an Image #####

img = cv2.imread('robot1.jpg')




######### Convert to GRAY ############
img1 =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



##### Histogram #####################
hist = cv2.calcHist(img1,[0], None, [256], [0,256])

plt.plot(hist, color ='k')
plt.show()


########################
cv2.imshow('main', img1)
cv2.waitKey(0)




##Exiting Function######

cv2.destroyAllWindows()
