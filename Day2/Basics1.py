#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 10:37:48 2020

@author: toshitt
"""
import imutils
import cv2
import os
import matplotlib.pyplot as plt
os.chdir("/home/toshitt/Desktop/Day2")



img= cv2.imread("elonmusk.jpg")

(h,w,d)=img.shape

print("width={},height={},depth={}".format(w,h,d))

plt.imshow(img)
plt.show()

#cv2.imshow("Dancing Elon",img)

##(B,G,R) = img[100,50]
##print("R={}, G={}, B={}".format(R,G,B))


#Slicing and cropping.

'''roi = img[150:450 ,800:999] #ROI stands for region of interest.
cv2.imshow("ROI",roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.waitKey(0)
cv2.destroyAllWindows()'''

#Resizing images

'''
r= 300/wPython 3.7.4 (default, Aug 13 2019, 20:35:49)
Type "copyright", "credits" or "license" for more information.

IPython 7.9.0 -- An enhanced Interactive Python.

Restarting kernel... 


 

In [1]: runfile('/home/toshitt/Desktop/pokedex-zernike/untitled0.py', wdir='/home/toshitt/Desktop/pokedex-zernike')
width=1080,height=911,depth=3

￼
dim=(300,int(h*r))
resized=cv2.resize(img,dim)
cv2.imshow("resized image",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
 '''

resized_new = imutils.resize(img,width=400)
cv2.imshow("Imutils resize",resized_new)
'''cv2.waitKey(0)
cv2.destroyAllWindows()'''
 
#rotating images
'''
center=(w//2,h//2)
#library function to rotate 
M=cv2.getRotationMatrix2D(center,-45,1.0)

rotated=cv2.warpAffine(M,(w,h))
cv2.imshow("Rotated image",resized_new)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
#blurred images
blurred = cv2.GaussianBlur(resized_new, (11,11), 0)
cv2.imshow("Blurred",blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
'''
#drawing rectangles

output = img.copy()
cv2.rectangle(output,(320,220),(540,270),(0,0,255),2)  #(cv2.rectangle(image_name,starting_pixel,ending_pixel,line color RGB format,line width)
cv2.imshow("Rectangled image",output)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#Drawing Lines
''''
output=img.copy()
cv2.line(output,(140,80),(240,160),(190,21,188),5)
cv2.imshow("Line",output)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''



