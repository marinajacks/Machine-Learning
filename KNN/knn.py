# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 22:02:31 2018

@author: Administrator
"""
import numpy as np

def loadDataSet(filename):
    dataSet = []
    labels = []
    fr = open(filename)
    lines=fr.readlines()
    for line in lines:
        lineArr = line.strip().split(',')
        value=[]
        for i in range(len(lineArr)-1):
            value.append(float(lineArr[i]))
        dataSet.append(value)
        labels.append([lineArr[-1]])   
    return dataSet , labels

#计算欧式距离
def calDist(X1 , X2):
    sum = 0
    for x1 , x2 in zip(X1 , X2):
        sum += (x1 - x2) ** 2
    return sum ** 0.5



def knn(data , dataSet , labels , k):
    n = np.shape(dataSet)[0]
    for i in range(n):
        dist = calDist(data , dataSet[i])
        #只记录两点之间的距离和已知点的类别
        labels[i].append(dist)
    #按照距离递增排序
    labels.sort(key=lambda x:x[1])
    count = {}
    #统计每个类别出现的频率
    for i in range(k):
        key = labels[i][0]
        if key in count:
            count[key] += 1
        else : count[key] = 1
    #按频率递减排序
    sortCount = sorted(count.items(),key=lambda item:item[1],reverse=True)
    return sortCount[0][0]#返回频率最高的key，即label



def main(path,path1):
    
    dataSet , labels = loadDataSet(path)
    testSet, Testlabels = loadDataSet(path1)
    newlabels=[]
    for i in range(len(testSet)):
        dataSet , labels = loadDataSet(path)
        label=knn(testSet[i], dataSet , labels , 10)
        newlabels.append(label)
    ps=percentage(Testlabels,newlabels)
    return newlabels,ps


def percentage(lables,lables1):
    length=len(lables)
    sum=0
    for i in range(len(lables)):
        if(lables[i][0]==lables1[i]):
            sum+=1
    return round(100*sum/length,2)


if __name__ == '__main__':
    p='E:\project\Machine-Learning\Kmeans\\trainiris.txt'
    p1='E:\project\Machine-Learning\Kmeans\\testtrain.txt'
    newlabels,ps=main(p,p1)
    for i in newlabels:
        print(i)
    print(ps)