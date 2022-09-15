# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# import torch
# import numpy as np
# import csv
# from sklearn import linear_model

# import matplotlib.pyplot as plt
# import torch
# import time


data = pd.read_csv("mnist_train.csv")

data = np.array(data)

m,n = data.shape

np.random.shuffle(data)

data_dev = data[0:1000].T
y_dev = data_dev[0]
x_dev = data_dev[1:n]

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n]

def init_params():
    w1 = np.random.rand(10,784)-0.5
    b1 = np.random.rand(10,1)-0.5
    
    w2 = np.random.rand(10,10)-0.5
    b2 = np.random.rand(10,1)-0.5
    return w1,b1,w2,b2

def forward_prop(w1,b1,w2,b2,x):
    z1 = w1.dot(x) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)

def ReLU_deriv(Z):
    return Z > 0

def relu(z):
    return np.maximum(0,z)

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))

def onehot(y):
    onehoty = np.zeros((y.size,y.max()+1))
    onehoty[np.arange(y.size),y] = 1
    return onehoty.T

def backwards_prop(z1,a1,z2,a2,w1,w2,x,y):
    onehoty = onehot(y)
    m = y.size
    dz2 = a2-onehoty
    dw2 = 1/m*dz2.dot(a1.T)
    db2 = 1/m*np.sum(dz2)
    
    dz1 = w2.T.dot(dz2) * ReLU_deriv(z1)
    dw1 = 1/m * dz1.dot(x.T)
    db1 = 1/m * np.sum(dz1)
    
    return dw1, db1, dw2, db2
    
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
    
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backwards_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


gradient_descent(x_train,y_train,0.1,10)
