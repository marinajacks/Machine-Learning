# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 23:17:26 2018

@author: 陈彪
版权所有,翻版必究
这边的函数主要是为了进行PCA算法的实现,在
此基础上实现对数据的降维,然后实现数据的分类,
测试数据集为IRIS数据集
"""

#coding=utf-8
import numpy as np

'''加载数据,主要是数据信息'''
def loadiris(p):
    f=open(p,'r')
    lines=f.readlines()
    datamat=[]
    for line in lines:
        a=[]
        for i in range(len(line.split(','))-1):
            a.append(float(line.split(',')[i]))
        datamat.append(a)
    return np.array(datamat)
'''获取数据集的标签信息'''       
def loadflags(p):
    f=open(p,'r')
    lines=f.readlines()
    flags=[]
    for line in lines:
        flags.append(line.strip().split(',')[-1])
    return flags

'''通过方差的百分比来计算将数据降到多少维是比较合适的，
函数传入的参数是特征值和百分比percentage，返回需要降到的维度数num'''
def eigValPct(eigVals,percentage):
    sortArray=np.sort(eigVals) #使用numpy中的sort()对特征值按照从小到大排序
    sortArray=sortArray[-1::-1] #特征值从大到小排序
    arraySum=sum(sortArray) #数据全部的方差arraySum
    tempSum=0
    num=0
    for i in sortArray:
        tempSum+=i
        num+=1
        if tempSum>=arraySum*percentage:
            return num

'''PCA的公式,主要是为了获取到训练数据的均值,特征向量信息,这就是训练学习过程'''
def pca(dataMat,percentage=0.9):
    meanVals=np.mean(dataMat,axis=0)  #对每一列求平均值，因为协方差的计算中需要减去均值
    meanRemoved=dataMat-meanVals
    covMat=np.cov(meanRemoved,rowvar=0)  #cov()计算方差
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))  #利用numpy中寻找特征值和特征向量的模块linalg中的eig()方法
    k=eigValPct(eigVals,percentage) #要达到方差的百分比percentage，需要前k个向量
    eigValInd=np.argsort(eigVals)  #对特征值eigVals从小到大排序
    eigValInd=eigValInd[:-(k+1):-1] #从排好序的特征值，从后往前取k个，这样就实现了特征值的从大到小排列
    redEigVects=eigVects[:,eigValInd]   #返回排序后特征值对应的特征向量redEigVects（主成分）    
    return np.mat(meanVals),redEigVects

'''求距离的公式,使用欧式距离的计算方法来计算距离'''
def diatance(vectors):
    sum=0
    for i in range(len(vectors)):
        sum+=pow(vectors[i],2)
    return np.sqrt(sum)
    

'''这部分是为了进行根据分类的结果进行划分的数据,返回了每一位数据的标签,均值和
训练得到的主成分矩阵'''
def train(p):
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
    
    meanVals=[]
    redEigVects=[]
    for i in range(len(datamat)):
        meanVal,redEigVect=pca(datamat[i],percentage=0.9)
        meanVals.append(meanVal)
        redEigVects.append(redEigVect)
    
    return names,meanVals,redEigVects

'''单条数据的判定结果,这里的单条数据是给定的一个数值'''
def test(p,x):
    distances=[] #存储距离信息,便于后期寻找
    x=np.mat(x).T
    names,meanVals,redEigVects=train(p)
    
    for i in range(len(redEigVects)):
        dist=redEigVects[i].dot((redEigVects[i].T).dot((x-meanVals[i].T)))
        distances.append(diatance(dist))
        
    J=0
    for i in range(len(distances)):
        if(distances[i]==min(distances)):
            J=i
    return names[J]


'''这个是计算的测试的准确度,'''
def accuracy(p,p1):
    initdata=loadiris(p1)
    initflag=loadflags(p1)
    
    testflag=[]
    for i in range(len(initdata)):
        testflag.append(test(p,initdata[i]))
        
    num=0
    right=0
    for i in range(len(initflag)):
        num+=1
        if(initflag[i]==testflag[i]):
            right+=1
    
    percent = str(round(100*right/num,2))+'%'
    return percent 
            
    
    
if __name__ == '__main__':
    p='E:\\project\\Machine-Learning\\PCA\\trainiris.txt'
    p1='E:\\project\\Machine-Learning\\PCA\\testtrain.txt'
    print(accuracy(p,p1))


        

    

    