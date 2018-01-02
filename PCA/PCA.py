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
给定的which不为0的时候'''
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
        
    
    
    
    



    
    