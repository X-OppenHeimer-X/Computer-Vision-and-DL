#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:22:52 2020

@author: toshitt
"""
#import libraries
import numpy as np
import cv2

#bring up labels
labels = ["Dog","cat","Donkey"]
#np.random.seed(1)
#iniitalise wwights
W= np.random.randn(3,3072)
b = np.random.randn(3)
#load our example image resize it, and then flatten it 
orig=cv2.imread("donkey.jpeg")
img=cv2.resize(orig,(32,32)).flatten()

print(img.shape)
#loop over the scores + labels amd display them
scores= W.dot(img) + b
#Draw the label with the highest score on the image as our
#prediction
for(label,score) in zip(labels,scores):
    print("[INFO] {}: {:.2f}".format(label,score))
    
cv2.putText(orig,"Label : {}".format(labels[np.argmax(scores)]),
            (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,128,128),2)
#dislpay image

cv2.imshow("Image",orig)
cv2.waitKey(0)
#