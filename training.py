# -*- coding: utf-8 -*-

import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
import time
import tkinter as tk

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# image_index = 7777 # You may select anything up to 60,000
# print(y_train[image_index]) # The label is 8
# plt.imshow(x_train[image_index], cmap='Greys')

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.tanh))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer = "adam",
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"]
              )

model.fit(x_train,y_train, epochs = 3, batch_size=32)

val_loss, val_acc = model.evaluate(x_test,y_test)

print(f"Loss is: {val_loss}, accuracy is: {val_acc}")

predicted = model.predict([x_test])

predicted = torch.Tensor(predicted)

predictions = torch.argmax(predicted, dim=1)

predictions = predictions.cpu().detach().numpy()

result = y_test-predictions

wrong = np.nonzero(result)[0]

print(f"Number of misidentifications: {str(len(wrong))} out of {str(len(y_test))}, accuracy: {str(1-(len(wrong)/len(y_test)))}")

import matplotlib, sys
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import pylab as plt
from scipy import ndimage
import tkinter as Tk

root = Tk.Tk()
root.wm_title("minimal example")

# image = plt.imread('test.jpg')
fig = plt.figure(figsize=(28,28))
im = plt.imshow(x_test[1], cmap='Greys') # later use a.set_data(new_data)
ax = plt.gca()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
# ax.set_xticklabels([]) 
# ax.set_yticklabels([]) 

# for index in wrong:
#     print(index)
#     print("Should be: " + str(y_test[index]))
#     print("Recognized as: " + str(predictions[index]))
#     plt.imshow(x_test[index], cmap='Greys')
#     plt.show()
#     time.sleep(2)
#     plt.clf()