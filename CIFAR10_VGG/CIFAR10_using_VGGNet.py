#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:51:09 2020

@author: toshitt
"""

import matplotlib 
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from CIFAR10_VGG import MiniVGGNet
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True,
                help="/home/CIFAR10_VGG")
args = vars(ap.parse_args())

print("[INFO] loading datasets .. .. ..")
((trainX,trainY),(testX,testY))= cifar10.load_data()

trainX = trainX.astype("float")/255.0
trainY = trainY.astype("float")/255.0

print("[INFO] compiling model  .. ..")

opt = SGD(lr=0.01,decay=0.01/40,momentum=0.9,nesterov = True)
model = MiniVGGNet.build(width=32,height=32,depth=3,classes=10)

model.compile(loss = "categorical_crossentropy",optimizer=opt,
              metrics=["accuracy"])

print("[INFO] training network .. . . .")

H= model.fit(trainX,trainY,validation_data=(testX,testY),
             batch_size=64,epochs=40,verbose=1)

print("[INFO] evaluating network ... . ..")
predictions=model.predict(testX,batch_size=64)
print(classification_report(testY.argmax(axis=1),prediction.argmax(axis=1),target_names= labelNames))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,40),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,40),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,40),H.history["val_accuracy"],label="val_acc")
plt.plot(np.arange(0,40),H.history["accuracy"],label="train_acc")

plt.title("training and loss accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
