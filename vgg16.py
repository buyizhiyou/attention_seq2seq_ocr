#-*-coding:utf8-*-

__author="buyizhiyou"
__date="2018-10-30"

'''
build vgg16 
'''
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras


def vgg16(x):

    with tf.variable_scope('vgg16',reuse=True):
        conv1 = keras.layers.Conv2D(64,3,activation='relu',padding='same')(x)
        conv2 = keras.layers.Conv2D(64,3,activation='relu',padding='same')(conv1)
        pool1 = keras.layers.MaxPool2D(padding='same')(conv2)

        conv3 = keras.layers.Conv2D(128,3,activation='relu',padding='same')(pool1)
        conv4 = keras.layers.Conv2D(128,3,activation='relu',padding='same')(conv3)
        pool2 = keras.layers.MaxPool2D(padding='same')(conv4)

        conv5 = keras.layers.Conv2D(256,3,activation='relu',padding='same')(pool2)
        conv6 = keras.layers.Conv2D(256,3,activation='relu',padding='same')(conv5)
        conv7 = keras.layers.Conv2D(256,3,activation='relu',padding='same')(conv6)
        pool3 = keras.layers.MaxPool2D(padding='same')(conv7)

        conv8 = keras.layers.Conv2D(512,3,activation='relu',padding='same')(pool3)
        conv9 = keras.layers.Conv2D(512,3,activation='relu',padding='same')(conv8)
        conv10 = keras.layers.Conv2D(512,3,activation='relu',padding='same')(conv9)
        pool4 = keras.layers.MaxPool2D(padding='same')(conv10)

        conv11 = keras.layers.Conv2D(512,3,activation='relu',padding='same')(pool4)
        conv12 = keras.layers.Conv2D(512,3,activation='relu',padding='same')(conv11)
        conv13 = keras.layers.Conv2D(512,3,activation='relu',padding='same')(conv12)
        pool5 = keras.layers.MaxPool2D(padding='same')(conv13)

    return pool5



if __name__ == "__main__":
    img=tf.placeholder(tf.float32,shape=(None,512,512,3))
    y = vgg16(img)
    x = np.random.random((1,512,512,3))
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        y = sess.run(y,feed_dict={img:x})
        print(y.shape)
