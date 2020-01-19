#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:56:03 2020

@author: toshitt
"""

import getPerspectiveTransform
import numpy as np
import argparse
import cv2

ap=argparse.ArgumentParser()

ap.add_argument("-i","--image_name",help="Path to image location")
ap.add_argument("-c","--coords",help="list of all coordinates")
args=vars(ap.parse_args())

image = cv2.imread(args["image_name"])
pts = np.array(eval(args["coords"]),dtype='float32')


warped = getPerspectiveTransform.four_point_transform(image,pts)

cv2.imshow("Orignal",image)
cv2.imshow("Warped",warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
