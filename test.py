#-*-coding:utf8-*-

__author = "buyizhiyou"
__date = "2018-8-14"

'''
test one image
'''

import skimage.io as io
from numpy import *
import pdb

import tensorflow as tf 
from im2str import *


inp = tf.placeholder(shape=[None, None, None, 1], dtype=tf.float32)
num_rows = tf.placeholder(tf.int32)
num_columns = tf.placeholder(tf.int32)
num_words = tf.placeholder(tf.int32)
true_labels = tf.placeholder(tf.int32, shape=[None, None])
_, (output, state) = build_model(inp,16, num_rows, num_columns, num_words)
pred = tf.to_int32(tf.argmax(output, 2))

pdb.set_trace()
img = io.imread('415_2.jpg', as_grey=True)
img = np.expand_dims(img,0)
img = np.expand_dims(img,3)
img = repeat(img, 16, axis=0)
words = len2chars[img.shape[2]]+2 #refer from img.width,add start and end
saver = tf.train.Saver()
with  tf.Session() as  sess:
    # saver = tf.train.import_meta_graph('saved_models/model-14-08-2018--19-24.meta')
    saver.restore(sess,'saved_models/model-14-08-2018--22-20')
    pred = sess.run(pred,feed_dict={inp: img,
                                    num_rows: img.shape[1],
                                    num_columns: img.shape[2],
                                    num_words: words})
    pred_str = []
    for i in pred:
        string = []
        for j in i:
            if j == 1:
                string.append(idx2vocab[j])
                break
            string.append(idx2vocab[j])
        pred_str.append(''.join(string))
    print("Result:",pred_str[0][5:-3])
