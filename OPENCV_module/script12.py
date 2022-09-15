### Import OpenCV & numpy #####

import cv2

import numpy as np

import matplotlib.pyplot as plt


#######################

#path = r"/"


### Read an Image #####

img = cv2.imread('blue.jpg')


#####  Equalize  #####################

channels = cv2.split(img)

eq_channel = []

for ch, color in zip(channels, ['B', 'G', 'R']):
    eq_channel.append(cv2.equalizeHist(ch))
eq_image = cv2.merge(eq_channel)














cv2.imwrite('equalize.jpg',eq_image)
### plot histogram ######
for i, col in enumerate(['b','g','r']):
    hist = cv2.calcHist([eq_image], [i], None, [256], [0,256])
    plt.plot(hist, color = col)
    plt.xlim([0,256])

plt.show()

resized1 = cv2.resize(img,(600,250),interpolation=cv2.INTER_AREA)
resized2 = cv2.resize(eq_image,(600,250),interpolation=cv2.INTER_AREA)

########################
cv2.imshow('main',resized2)
cv2.waitKey(0)

cv2.imshow('main',resized2)
cv2.waitKey(0)



##Exiting Function######

cv2.destroyAllWindows