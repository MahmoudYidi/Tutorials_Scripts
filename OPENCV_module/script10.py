### Import OpenCV & numpy #####

import cv2

import numpy as np

import matplotlib.pyplot as plt


#######################

#path = r"/"


### Read an Image #####

img = cv2.imread('snow.jpg')


##### Histogram #####################
for i, col in enumerate(['b', 'g', 'r']):
    hist = cv2.calcHist([img], [i], None, [256], [0,256])
    plt.plot(hist, color = col)
    plt.xlim([0,256])

plt.show()
cv2.waitKey(0)



########################
cv2.imshow('main', img)
cv2.waitKey(0)




##Exiting Function######

cv2.destroyAllWindows()
