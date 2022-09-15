### Import OpenCV & numpy #####

import cv2

import numpy as np

import matplotlib.pyplot as plt


### Read an Image #####

img = cv2.imread('sudoku.PNG')

img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



#####  HOUGH LINE #####################

edge = cv2.Canny(img1, 15, 450, apertureSize=3)

lines = cv2.HoughLines(edge, 1, np.pi/180, 200)

for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 500*(-b))
        y1 = int(y0 + 500*(a))
        x2 = int(x0 - 500*(-b))
        y2 = int(y0 - 500*(a))
        cv2.line(img,(x1,y1), (x2,y2), (0,0,255), 2)






########################
cv2.imshow('main', img)
cv2.waitKey(0)




##Exiting Function######

cv2.destroyAllWindows