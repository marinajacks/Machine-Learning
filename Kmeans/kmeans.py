# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 21:00:04 2018

@author: Administrator
"""

#################################################
# kmeans: k-means cluster
# Author : wangpeifen
# Date   : 2013-12-25
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

import numpy as np
import time
import matplotlib.pyplot as plt
import random

# calculate Euclidean distance
def euclDistance(vector1, vector2):
	return np.sqrt(sum(np.power(vector2 - vector1, 2)))

def euclDistance1(vector1, vector2):
    sum=0
    for i in range(len(vector1)):
        sum+=np.power(vector1[i]-vector2[i],2)
    return np.sqrt(sum)
        
	#return np.sqrt(sum(np.power(vector2 - vector1, 2)))

# init centroids with random samples
'''这个函数用来计算分类的中心点信息'''
def initCentroids(dataSet, k):
	numSamples, dim = dataSet.shape
	centroids = np.zeros((k, dim))
	for i in range(k):
		index = int(random.uniform(0, numSamples))
		centroids[i, :] = dataSet[index, :]
	return centroids

# k-means cluster
def kmeans(dataSet, k):
	numSamples = dataSet.shape[0]
	# first column stores which cluster this sample belongs to,
	# second column stores the error between this sample and its centroid
	clusterAssment = np.mat(np.zeros((numSamples, 2)))
	clusterChanged = True

	## step 1: init centroids
	centroids = initCentroids(dataSet, k)

	while clusterChanged:
		clusterChanged = False
		## for each sample
		for i in  range(numSamples):
			minDist  = 100000.0
			minIndex = 0
			## for each centroid
			## step 2: find the centroid who is closest
			for j in range(k):
				distance = euclDistance(centroids[j, :], dataSet[i, :].astype('float64')  )
				if distance < minDist:
					minDist  = distance
					minIndex = j
			
			## step 3: update its cluster
			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True
				clusterAssment[i, :] = minIndex, minDist**2

		## step 4: update centroids
		for j in range(k):
			pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
			centroids[j, :] = np.mean(pointsInCluster.astype('float64'), axis = 0)

	print ('Congratulations, cluster complete!')
	return centroids, clusterAssment


    


import numpy as np
import time
import matplotlib.pyplot as plt

## step 1: load data
print ("step 1: load data...")
dataSet = []
dataflag=[]
path='E:\project\Machine-Learning\Kmeans\\iris.txt'
f = open(path)
lines = f.readlines()
for line in lines:
    lineArr = line.strip().split(',')
    value=[]
    for i in range(len(lineArr)-1):
        value.append(float(lineArr[i]))
    dataSet.append(value)
    dataflag.append(lineArr[-1])        
   

## step 2: clustering...
print ("step 2: clustering...")
dataSet = np.mat(dataSet)
k = 3
centroids, clusterAssment = kmeans(dataSet, k)

