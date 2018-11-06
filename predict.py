#-*-coding:utf8-*-

__author = "buyizhiyou"
__date = "2018-11-5"

import numpy as np
import tensorflow as tf
from skimage import io

from config import config
from model import seq2seq
from preprocess import batchify, load_data
from vgg16 import vgg16
from train import idx2vocab

batch_size = config.batch_size

# with tf.Session() as sess:
#     model_file = tf.train.latest_checkpoint(config.checkpoint_path)
#     saver = tf.train.import_meta_graph('checkpoint/models.ckpt-0.meta')#build model from .meta file
#     saver.restore(sess,model_file)

#     im = io.imread('data/1695_0.jpg')
#     # len = img2len[im.shape]
#     imgs = np.expand_dims(im,0).astype(np.float32)
#     imgs = np.repeat(imgs,batch_size,axis=0)
#     labels =np.expand_dims(([0]*30)+[1],axis=0)#according to img's length ,we can get the text length
#     val_true_labels = np.repeat(labels,batch_size,axis=0)
#     graph = tf.get_default_graph()
#     pred = graph.get_operation_by_name('trained_1/pred_outputs_sq')
#     #[n.name for n in tf.get_default_graph().as_graph_def().node]
#     #graph.get_tensor_by_name()
#     #pred = tf.get_collection("pred")[0]
#     img = graph.get_operation_by_name('Placeholder').outputs[0]
#     true_labels = graph.get_operation_by_name('Placeholder_1').outputs[0]

#     train_outputs = sess.run(pred,feed_dict={img:imgs,true_labels:val_true_labels})
#     print(train_outputs)


img = tf.placeholder(shape=[None,None,None,3],dtype=tf.float32)
true_labels = tf.placeholder(tf.int32,shape=[None,None])

#build network and train_step
conv = vgg16(img)
encoder_inputs = conv[:,0,:,:]
train_outputs,weights = seq2seq(encoder_inputs,true_labels,'train')
pred_outputs = seq2seq(encoder_inputs,true_labels,'infer')
model_file = tf.train.latest_checkpoint(config.checkpoint_path)#restore model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if model_file:
        print("model restored from {}".format(model_file))
        saver.restore(sess,model_file)

    im = io.imread('data/1695_0.jpg') 
    imgs = np.expand_dims(im,0).astype(np.float32)
    imgs = np.repeat(imgs,batch_size,axis=0)
    labels =np.expand_dims(([0]*30)+[1],axis=0)#according to img's length ,we can get the text length
    val_true_labels = np.repeat(labels,batch_size,axis=0)

    pred_outputs1 = sess.run([pred_outputs],feed_dict={img:imgs,true_labels:val_true_labels})
    ids = pred_outputs1[0].sample_id[0]
    strs = ''.join([idx2vocab[id] for id in ids][:-1])
    print(strs)
