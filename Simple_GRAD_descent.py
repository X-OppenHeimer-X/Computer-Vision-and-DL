#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:55:12 2020

@author: toshitt
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    
    return x*(1-x)

def predict(X,W):
    
    preds = sigmoid_activation(X.dot(W))
    
    preds[preds<=0.5]=0
    
    preds[preds>0.5]=1 
    
    return preds

ap = argparse.ArgumentParser()

ap.add_argument("-e","--epochs",type=float,default=200,
                help=200)
ap.add_argument("-a","--alpha",type=float,default=0.086,
                help=0.068)
args=vars(ap.parse_args())

(X,y) = make_blobs(n_samples=1000,n_features=2,centers=2,
        cluster_std=1.5,random_state=1)
y=y.reshape((y.shape[0],1))

X = np.c_[X,np.ones((X.shape[0]))]
(trainX,testX,trainY,testY)= train_test_split(X,y,test_size=0.25,
                                random_state=42)

print("[INFO] training .....")

W = np.random.randn(X.shape[1],1)
losses=[]

for epoch in np.arange(0,args["epochs"]):
    
    preds = sigmoid_activation(trainX.dot(W))
    
    errors= preds-trainY
    loss = np.sum(errors**2)
    losses.append(loss)
    d = errors*sigmoid_deriv(preds)
    gradient = (trainX.T.dot(d))
    
    W += -args["alpha"] * gradient
    
    if epoch == 0 or (epoch+1)%5==0:
        print("[INFO] epoch ={},loss ={:.7f}".format(int(epoch+1),loss))
       
print("[INFO] evaluating...")
preds = predict(testX,W)

print(classification_report(testY,preds)) 

#data classification plot
plt.style.use("ggplot")
plt.figure
plt.title("Data") 

plt.scatter(testX[:,0],testX[:,1],marker="o",c=testY[:,0],s=30)

#construct a figure that plots the loss over time


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,args["epochs"]),losses)
plt.title("Training loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")

plt.show()         
     
        