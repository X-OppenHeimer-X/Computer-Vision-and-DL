#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Aim : to create a persepective
"""
Created on Sat Jan 18 14:32:53 2020

@author: toshitt
"""

import cv2
import numpy as np

def order_points(pts):
    #initialise set of 4 points that will be ordered
    #such that the first entry in the list is on top left
    #second->top right, third->bottom right
    #fourth->bottom left
    
    rect = np.zeros((4,2),dtype="float32")
    #the top left will have the lowest sum
    #the bottom right will have largest sum
    s=pts.sum(axis=1)
    rect[0]=pts[np.argmin(s)]
    rect[3]=pts[np.argmax(s)]
    
    return rect

def four_point_transform(image,points):
    #obtain a consistent order of the points and unpack them.
    #individually
    
    rect= order_points(points)
    (tl,tr,br,bl)=rect
    
    widthA = np.sqrt((br[0]-bl[0])**2+(br[1]-bl[0])**2)
    widthB = np.sqrt((tr[0]-tl[0])**2+(tr[1]-tl[1])**2)
    maxWidth = max(int(widthA),int(widthB))
    
    heightA = np.sqrt((tl[0]-bl[0])**2+(tl[1]-bl[0])**2)
    heightB = np.sqrt((tr[0]-br[0])**2+(tr[1]-br[0])**2)
    maxHeight=max(int(heightA),int(heightB))
    
    #construction the set of teh destination points to obtain
    dst=np.array([
            [0,0],
            [maxWidth-1,0],
            [maxWidth-1,maxHeight-1],
            [0,maxHeight-1]])
    M=cv2.getPerspectiveTransform(rect, dst)
    warped=cv2.warpPerspective(image,M,(maxWidth,maxHeight))
    
    return warped

