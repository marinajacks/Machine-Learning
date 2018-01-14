# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 14:04:01 2018

@author: 陈彪
版权所有,翻版必究
"""

'''
    使用python解析二进制文件
'''

import numpy as np
import struct
'''函数的标签给定的是默认的为0,也就是训练数据集,但实际上还有测试数据集,测试数据集就是
给定的which不为0的时候,现在的情况是,给出的数据是这样的,不能实现的是直接进行相关的计算
好的处理方式是,在每次结束的时候,直接将数据弄成行的格式即可'''
def loadImageSet(which=0):
    print ("load image set")
    binfile=None
    if which==0:
        binfile = open("D:\\Projects\\train-images.idx3-ubyte", 'rb')
    else:
        binfile=  open("D:\\Projects\\t10k-images.idx3-ubyte", 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII' , buffers ,0)
    print ("head,"),head

    offset=struct.calcsize('>IIII')
    imgNum=head[1]
    width=head[2]
    height=head[3]
    #[60000]*28*28
    bits=imgNum*width*height
    bitsString='>'+str(bits)+'B' #like '>47040000B'

    imgs=struct.unpack_from(bitsString,buffers,offset)

    binfile.close()
    imgs=np.reshape(imgs,[imgNum,width,height])
    
    imgg=[]
    for i in range(len(imgs)):
        imgg.append(imgs[i].reshape([1,width*height])[0])
    imgs=imgg
    
    print ("load imgs finished")
    return imgs



def loadLabelSet(which=0):
    print ("load label set")
    binfile=None
    if which==0:
        binfile = open("D:\\Projects\\train-labels.idx1-ubyte", 'rb')
    else:
        binfile=  open("D:\\Projects\\t10k-labels.idx1-ubyte", 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II' , buffers ,0)
    print ("head,"),head
    imgNum=head[1]

    offset = struct.calcsize('>II')
    numString='>'+str(imgNum)+"B"
    labels= struct.unpack_from(numString , buffers , offset)
    binfile.close()
    labels=np.reshape(labels,[imgNum,1])

    #print labels
    print ('load label finished')
    return labels


if __name__=="__main__":
    imgs=loadImageSet()
    #import PlotUtil as pu
    #pu.showImgMatrix(imgs[0])
    labels=loadLabelSet()
    
    testimgs=loadImageSet(which=1)
    #import PlotUtil as pu
    #pu.showImgMatrix(imgs[0])
    testlabels=loadLabelSet(which=1)
    
p1='E:\project\Machine-Learning\PCA\Mnist\\5.txt'
f=open(p1,'w')

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
def pca(dataMat,percentage=0.95):
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
    
    
def train():
    datamats=loadImageSet()
    flags=loadLabelSet()
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
#实际上这里给的数据是不正常的,正确的数据存储是按照列存储的,现在的情况是
#按照行存储的,这个数据需要调整才可以
    meanVals=[]
    redEigVects=[]
    for i in range(len(datamat)):
        meanVal,redEigVect=pca(np.mat(datamat[i]),percentage=0.95)
        meanVals.append(meanVal)
        redEigVects.append(redEigVect)
    
    return names,meanVals,redEigVects    
  


dataMat=np.mat(datamat[0])
imgss=dataMat











p1='E:\project\Machine-Learning\PCA\Mnist\\datass1.txt'
f=open(p1,'w')
for i in range(len(imgss)):
    for j in range(len(imgss[i])):
        f.write(str(imgss[i][j])+' ')
    f.write('\n')
f.close()


    
    
def test(names,meanVals,redEigVects,x):
   # distances=[] #存储距离信息,便于后期寻找
    testflags=[]
    for j in range(len(testlabels)):
        distances=[]
        x=np.mat(testimgs[j]).T
        #names,meanVals,redEigVects=train()
        
        for i in range(len(redEigVects)):

            dist=((redEigVects[i].T).dot((x-meanVals[i].T)))#+meanVals[i].T
            #dist=redEigVects[i].dot((redEigVects[i].T).dot((x-meanVals[i].T)))
            '''这里实际上是有两个问题的 '''
            #+meanVals[i].T
            distances.append(diatance(dist))
            
        J=0
        for i in range(len(distances)):
            if(distances[i]==min(distances)):
                J=i
        testflags.append(names[J])
    #names[J]
    return names[J]
    
    


testlabels[0:9]

test=testflags
real=testlabels

def right(test,real):
    length=len(real)
    
    sum=0
    for i in range(len(real)):
        if(test[i]==real[i]):
            sum=sum+1
    
    return np.round(sum/length,2)
    
    
    

def  test(i):
    print(' if(labels[i]=='+str(i)+'):')
    print('\tlabels'+str(i)+'.append(i)')
    







def discriminant():
    #这个函数主要是为了判别数据的归属,为后续的数据进行判别
    
    
    
    
    
    '''下面给出的是计算每个labels对应的'''
    labels0=[]
    labels1=[]
    labels2=[]
    labels3=[]
    labels4=[]
    labels5=[]
    labels6=[]
    labels7=[]
    labels8=[]
    labels9=[]
    for i in range(len(labels)):
         if(labels[i]==0):
            labels0.append(i)
         if(labels[i]==1):
                labels1.append(i)
         if(labels[i]==2):
                labels2.append(i)
         if(labels[i]==3):
                labels3.append(i)
         if(labels[i]==4):
                labels4.append(i)
         if(labels[i]==5):
                labels5.append(i)
         if(labels[i]==6):
                labels6.append(i)
         if(labels[i]==7):
                labels7.append(i)
         if(labels[i]==8):
                labels8.append(i)
         if(labels[i]==9):
                labels9.append(i)
            
def means(imgs,labels0):
    #这个函数计算均值
    img=[]
    for i in range(len(labels0)):
        img.append(np.reshape(imgs[labels0[i]],[28*28,1]))
        
    a=[]
    
    #下面主要要做的就是计算这个均值向量
    for i in range(len(img[0])):
        sum=0
        for j in range(len(img)):
            sum+=img[j][i]
        a.append(np.round(sum/len(img),2))
    return a

        
def covs(labels0,img):
    #这个函数主要是为了计算协方差矩阵用的,注意这里处理的时候需要注意
    #这里计算的是特征值和特征向量,下面计算得到的e和v分别是这两个数据
    a=means(img)
    b=[]
    for i in range(len(labels0)):
        b.append(np.reshape(imgs[labels0[i]],[28*28,1]))
        
    mats=0
    for i in range(len(b)):
        mats+=((b[i]-a).dot((b[i]-a).T))/(len(labels0)-1)
    #下面计算特征值和特征向量
    e,v=np.linalg.eig(mats)
    A1=np.linalg.eigvals(mats)
    
    
    
    
#这部分实际上是为了获取特征值的实数部分,然后再进行下一步计算

p="d://projects//1.txt"
f=open(p,'w')
s1=[]
for i in range(len(e)):
    a=0
    if(e[i].real<pow(10,-3)):
        a=0
    else:
        a=e[i].real
        
    f.write(str(a)+'\t'+str(e[i])+'\n')
f.close()

#这个函数是为了获取主成分,这里的函数主要是计算主成分的,
#输入的结果是
def getPrincipal(e):
    eigs=[]
    for i in range(len(e)):
        a=0
        if(e[i].real<pow(10,-3)):
            a=0
        else:
            a=e[i].real
        eigs.append(round(float(a),2))
    
    I=0    #计算计算迭代的次数,这个次数实际上就是主成分的个数
    sum=0  #计算总的特征值
    for i in range(len(eigs)):
        sum=sum+eigs[i]
    sum=int(sum)
    sums=0       #计算累加的数据
    
    while(1):
        sums=sums+eigs[I]
        if(sums/sum>0.8):
            break
        I=I+1
    #这里计算出来的I实际上就是主成分发的个数,陆老师给出的数据
    #是25,这里计算得到的数据是26,其实是一样的.
    
        
    
    
        
    
    

'''下面还要计算每个变量到这个10个训练好的机器之间的距离,然后判定这个
测试数据到底属于哪一个标签,判断所属的分类实际上是计算测试点到机器之间
的距离,这样的话才能计算出最终的结果'''





def  distances(labels,tlabels):
    #这个函数主要是为了进行计算函数测试的正确率
    right=0
    for i in range(len(labels)):
        if(labels[i]==tlabels[i]):
            right=right+1
    return np.round(right/len(labels),2)
        
    




import pandas as pd
import os

path="E://迅雷下载//IRIS.csv"

pwd = os.getcwd()
os.chdir(os.path.dirname(path))
df = pd.read_csv(os.path.basename(path),header=None)
os.chdir(pwd)




import pandas as pd
path="E://迅雷下载//IRIS.csv"
df=pd.read_csv(path)
    
    

'''下面对IRIS数据进行处理,得到想要的结果

'''
    

import numpy as np
x= np.mat([[4,-1,-1,-1,-1,0,0,0],[-1,3,-1,0,0,-1,0,0],[-1,-1,3,-1,0,0,0,0],[-1,0,-1,3,0,0,0,-1],[-1,0,0,0,2,0,0,-1],[0,-1,0,0,0,2,-1,0],[0,0,0,0,0,-1,2,-1],[0,0,0,-1,-1,0,-1,3]])
a,b=np.linalg.eig(x)



    
    