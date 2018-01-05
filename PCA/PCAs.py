# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 15:52:12 2018

@author: 陈彪
版权所有,翻版必究
这边的函数主要是为了进行PCA算法的实现
"""
#coding=utf-8
from numpy import *

import matplotlib.pyplot as plt

#p='E:\project\Machine-Learning\PCA\iris.txt'
def loadiris(p):
    f=open(p,'r')
    lines=f.readlines()
    datamat=[]
    for line in lines:
        a=[]
        for i in range(len(line.split(','))-1):
            a.append(float(line.split(',')[i]))
        datamat.append(a)
    return array(datamat)
        
def loadflags(p):
    f=open(p,'r')
    lines=f.readlines()
    flags=[]
    for line in lines:
        flags.append(line.strip().split(',')[-1])
    return flags


#这部分是为了进行根据分类的结果进行划分的数据,返回了每一位数据的
def getdatamat(p):
    datamats=loadiris(p)
    flags=loadflags(p)
    names=[flags[0]]
    
    
    for i in range(len(flags)):
        if(flags[i] in names):
            pass
        else:
            names.append(flags[i])
    address=[]
    datamat=[]
    for i in range(len(names)):
        address.append([])
        datamat.append([])
        
    for i in range(len(flags)):
        for j in range(len(names)):
            if(flags[i]==names[j]):
                address[j].append(i)
                datamat[j].append(datamats[i])
                
    return names,address,datamat


    













'''通过方差的百分比来计算将数据降到多少维是比较合适的，
函数传入的参数是特征值和百分比percentage，返回需要降到的维度数num'''
def eigValPct(eigVals,percentage):
    sortArray=sort(eigVals) #使用numpy中的sort()对特征值按照从小到大排序
    sortArray=sortArray[-1::-1] #特征值从大到小排序
    arraySum=sum(sortArray) #数据全部的方差arraySum
    tempSum=0
    num=0
    for i in sortArray:
        tempSum+=i
        num+=1
        if tempSum>=arraySum*percentage:
            return num

'''pca函数有两个参数，其中dataMat是已经转换成矩阵matrix形式的数据集，列表示特征；
其中的percentage表示取前多少个特征需要达到的方差占比，默认为0.9'''


'''
def pca(dataMat,percentage=0.99):
    meanVals=mean(dataMat,axis=0)  #对每一列求平均值，因为协方差的计算中需要减去均值
    meanRemoved=dataMat-meanVals
    covMat=cov(meanRemoved,rowvar=0)  #cov()计算方差
    eigVals,eigVects=linalg.eig(mat(covMat))  #利用numpy中寻找特征值和特征向量的模块linalg中的eig()方法
    k=eigValPct(eigVals,percentage) #要达到方差的百分比percentage，需要前k个向量
    eigValInd=argsort(eigVals)  #对特征值eigVals从小到大排序
    eigValInd=eigValInd[:-(k+1):-1] #从排好序的特征值，从后往前取k个，这样就实现了特征值的从大到小排列
    redEigVects=eigVects[:,eigValInd]   #返回排序后特征值对应的特征向量redEigVects（主成分）
    lowDDataMat=meanRemoved*redEigVects #将原始数据投影到主成分上得到新的低维数据lowDDataMat
    reconMat=(lowDDataMat*redEigVects.T)+meanVals   #得到重构数据reconMat
    return lowDDataMat,reconMat
'''


def pcas(dataMat,percentage=0.99):
    meanVals=mean(dataMat,axis=0)  #对每一列求平均值，因为协方差的计算中需要减去均值
    meanRemoved=dataMat-meanVals
    covMat=cov(meanRemoved,rowvar=0)  #cov()计算方差
    eigVals,eigVects=linalg.eig(mat(covMat))  #利用numpy中寻找特征值和特征向量的模块linalg中的eig()方法
    k=eigValPct(eigVals,percentage) #要达到方差的百分比percentage，需要前k个向量
    eigValInd=argsort(eigVals)  #对特征值eigVals从小到大排序
    eigValInd=eigValInd[:-(k+1):-1] #从排好序的特征值，从后往前取k个，这样就实现了特征值的从大到小排列
    redEigVects=eigVects[:,eigValInd]   #返回排序后特征值对应的特征向量redEigVects（主成分）    
    return mat(meanVals),redEigVects

#求距离的公式
def diatance(vectors):
    sum=0
    for i in range(len(vectors)):
        sum+=pow(vectors[i],2)
    return sqrt(sum)
    


    
def main():
    distances=[]
    
    #x=mat([4.5,2.3,1.3,0.5]).T
    x=mat([6.4,3.2,5.3,2.5]).T

    for i in range(len(datamat)):
        meanVals,redEigVects=pcas(datamat[i],percentage=0.9)
        dist=redEigVects.dot((redEigVects.T).dot((x-meanVals.T)))
        distances.append(diatance(dist))
    

    distances
    
    J=0
    for i in range(len(distances)):
        if(distances[i]==min(distances)):
            J=i
    names[J]
        

    

    