# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 12:50:07 2018

@author: Administrator
这个是自然语言处理学习的一个简单的软件的实现,记录了相关的一些信息,主要学习中使用到
的自然语言处理的信息
"""

def test():
    import nltk


    sentence = "At eight o'clock on Thursday morning"
    
    
    tokens = nltk.word_tokenize(sentence)
    
    tokens
    ['At', 'eight', "o'clock", 'on', 'Thursday', 'morning',
    'Arthur', 'did', "n't", 'feel', 'very', 'good', '.']
    tagged = nltk.pos_tag(tokens)
    tagged[0:6]
    [('At', 'IN'), ('eight', 'CD'), ("o'clock", 'JJ'), ('on', 'IN'),
    ('Thursday', 'NNP'), ('morning', 'NN')]
    
    
    import requests as r
    url='http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris=r.get(url)


if __name__ == "__main__":
    print('hello world')