# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 12:03:18 2018

@author: Administrator
"""

# Author：Charlotte
# Plot mnist dataset
from keras.datasets import mnist
import matplotlib.pyplot as plt
# load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('PuBuGn_r'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('PuBuGn_r'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('PuBuGn_r'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('PuBuGn_r'))
# show the plot
plt.show()