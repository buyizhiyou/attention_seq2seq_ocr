#-*-coding:utf8-*-

__author="buyizhiyou"
__date="2018-10-30"

import sys,os
import random
import numpy as np
from skimage import io
import tensorflow as tf 


vocab = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k',
        'l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G',
        'H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','/','_','#',' ']

vocab2idx = dict([(vocab[i], i+4) for i in range(len(vocab))])
vocab2idx['START']=0
vocab2idx['END']=1
vocab2idx['UNKNOWN']=2
vocab2idx['PADDING']=3
idx2vocab = dict([(i+4,vocab[i]) for i in range(len(vocab))])
idx2vocab[0]='START'
idx2vocab[1]='END'
idx2vocab[2]='UNKNOWN'
idx2vocab[3]='PADDING'


def load_data(path):
    def labels_to_vector(labels):
        # res=[0]#'START'
        res=[]
        for token in labels:
            if token in vocab:
                res.append(vocab2idx[token])
            else:
                res.append(2)#'UNKNOWN'
        res.append(1)#'END'

        return res
    
    train = open(path).read().split('\n')[:-1]
    random.shuffle(train)
    def read_image(line):
        cols = line.split('*')
        img = io.imread('data/sample/'+cols[0])
        return (img,labels_to_vector(cols[1]))
    
    train = list(map(read_image,train))

    return train

def batchify(data,batch_size):
    #group by image size
    res = {}
    for datum in data:
        if datum[0].shape not in res:
            res[datum[0].shape]=[datum]
        else:
            res[datum[0].shape].append(datum)
    for size in res.keys():
        group = sorted(res[size],key=lambda x:len(x[1]))
        for i in range(0,len(group),batch_size):
            # import pdb; pdb.set_trace()
            images = list(map(lambda x:np.expand_dims(x[0],0),group[i:i+batch_size]))
            batch_images = np.concatenate(images,0)
            seq_len = max([len(x[1]) for x in group[i:i+batch_size]])

            def preprocess(x):
                arr = np.array(x[1])
                pad = np.pad(arr,(0,seq_len-arr.shape[0]),'constant',constant_values=3)
                return np.expand_dims(pad,0)
            labels = list(map(preprocess,group[i:i+batch_size]))
            batch_labels = np.concatenate(labels,0)
            if batch_images.shape[0]!=batch_size:
                continue
            batch = (batch_images,batch_labels)
            
            yield batch




    
