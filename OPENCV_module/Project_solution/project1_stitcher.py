

import cv2
import numpy as np

#Read images ########################################################################
img2 =cv2.imread('IMG_1879.jpg')
img1 = cv2.imread('IMG_1880.jpg')

#img1 =cv2.imread('IMG_1884.jpg')
#img2 = cv2.imread('IMG_1885.jpg')

#img1 =cv2.imread('result.jpg')
#img2 = cv2.imread('IMG_1886.jpg')


img1 = cv2.resize(img1,(1000,1000),interpolation=cv2.INTER_AREA)

img2 = cv2.resize(img2,(1000,1000),interpolation=cv2.INTER_AREA)

########################################################################################

images= []

images.append(img1)

images.append(img2)

stitcher=cv2.Stitcher_create()
(status, stitched)=stitcher.stitch(images)


########################################################################################

cv2.imshow('result', stitched)
cv2.waitKey(0)

cv2.destroyAllWindows()




