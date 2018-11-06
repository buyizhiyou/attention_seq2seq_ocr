#-*-coding:utf8-*-

__author = "buyizhiyou"
__date = "2018-11-5"

import numpy as np
import tensorflow as tf
from skimage import io

from config import config

batch_size = config.batch_size

model_file = tf.train.latest_checkpoint(config.checkpoint_path)
with tf.Session() as sess:

    saver = tf.train.import_meta_graph('checkpoint/models.ckpt-0.meta')
    saver.restore(sess,model_file)
    imgs = np.expand_dims(io.imread('data/1695_0.jpg'),0).astype(np.float32)
    imgs = np.repeat(imgs,batch_size,axis=0)
    labels =np.expand_dims(list(range(17)),axis=0)#according to img's length ,we can get the text length
    labels = np.repeat(labels,batch_size,axis=0)
    graph = tf.get_default_graph()
    pred = graph.get_operation_by_name('trained/pred_outputs_sq')
    # pred = tf.get_collection("pred")[0]
    img = graph.get_operation_by_name('Placeholder').outputs[0]
    true_labels = graph.get_operation_by_name('Placeholder_1').outputs[0]
    train_outputs = sess.run(pred,feed_dict={img:imgs,true_labels:labels})
    print(train_outputs)
