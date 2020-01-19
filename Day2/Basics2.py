#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:20:25 2020

@author: toshitt
"""

import cv2
import matplotlib.pyplot as plt
import os
import imutils

print(os.getcwd())

img=cv2.imread("elonmusk.jpg")
plt.imshow(img)
plt.show()

new_image=imutils.resize(img,width=570)

#converting image to grey scale
gray=cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image",gray)
cv2.destroyAllWindows()



#Edge Detection
'''
edged=cv2.Canny(gray,150,140)
cv2.imshow("Edged",edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#Image Thresholding
''''
thresh = cv2.threshold(gray,64,105,cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh",thresh)
cv2.waitKey(0)'''