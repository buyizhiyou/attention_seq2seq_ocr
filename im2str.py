#-*-coding:utf8-*-

__date__ = "2017-11-5"
__author__ = "buyizhiyou"

import random, time, os, decoder
import numpy as np
import skimage.io as io
import tensorflow as tf
import pdb
import Levenshtein
from skimage.filters import threshold_otsu

from tensorflow.contrib.legacy_seq2seq import embedding_attention_seq2seq,embedding_attention_decoder
from tensorflow.python import shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell,rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

from decoder2 import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"#choose GPU 1

vocab = open('data/vocab.txt').read().split('\n')
vocab = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k',
                'l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

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

chars2len = {4: 80, 5: 97, 6: 115, 7: 132, 8: 150}
len2chars = {80: 4, 97: 5, 115: 6, 132: 7, 150: 8}#by gen_data.py,we get the map of  words's numbers and sample image width


# four meta keywords
# 0: START
# 1: END
# 2: UNKNOWN
# 3: PADDING

def load_data():

    def labels_to_vector(labels):
        res = [0]#append 'start'
        for token in labels:
            if token in vocab:
                res.append(vocab2idx[token])
            else:
                res.append(2)#append 'unknown'
        res.append(1)#append 'end'
        return res

    train = open('data/train.txt').read().split('\n')[:-1]
    val = open('data/val.txt').read().split('\n')[:-1]
    def import_images(line):
        cols = line.split(' ')
        img = io.imread('data/sample/'+cols[0],as_grey=True)

        return (img, (labels_to_vector(cols[1])))

    train = list(map(import_images, train))
    val = list(map(import_images, val))

    return train, val

def batchify(data, batch_size):
    # group by image size
    res = {}
    for datum in data:
        if datum[0].shape not in res:
            res[datum[0].shape] = [datum]
        else:
            res[datum[0].shape].append(datum)
    batches = []
    for size in res.keys():#[(40,80),...,(40,150)],5 bucket
        # batch by similar sequence length within each image-size group -- this keeps padding to a
        # minimum
        group = sorted(res[size], key= lambda x: len(x[1]))
        for i in range(0, len(group), batch_size):
            images = list(map(lambda x: np.expand_dims(np.expand_dims(x[0],0),3), group[i:i+batch_size]))
            batch_images = np.concatenate(images, 0)#(batch_size,h,w,c)
            seq_len = max([ len(x[1]) for x in group[i:i+batch_size]])
            def preprocess(x):
                arr = np.array(x[1])
                pad = np.pad( arr, (0, seq_len - arr.shape[0]), 'constant', constant_values = 3)
                return np.expand_dims( pad, 0)
            labels = list(map( preprocess, group[i:i+batch_size]))#padding
            batch_labels = np.concatenate(labels, 0)#(batch_size,max_length)
            
            batches.append( (batch_images, batch_labels) )
    #skip the last incomplete batch for now
    return batches

layer_params = [ [ 64,  3, 'same',  'conv1', False], 
                 [ 128, 3, 'same',  'conv2', False],
                 [ 256, 3, 'same',  'conv3', False],
                 [ 256, 3, 'same',  'conv4', False],
                 [ 512, 3, 'same',  'conv5', True], 
                 [ 512, 3, 'same',  'conv6', True],
                 [ 512, 3, 'valid',  'conv7', False],]
def init_cnn(inp,training=True):
    def norm_layer(bottom, training, name):
        """Short function to build a batch normalization layer with less syntax"""
        top = tf.layers.batch_normalization(bottom, axis=3,  # channels last,
                                            training=training,
                                            name=name)
        return top
    def conv_layer(bottom, params, training):
        """Build a convolutional layer using entry from layer_params)"""
        batch_norm = params[4]# Boolean

        if batch_norm:
            activation = None
        else:
            activation = tf.nn.relu

        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer(value=0.0)

        top = tf.layers.conv2d(bottom,
                                filters=params[0],
                                kernel_size=params[1],
                                padding=params[2],
                                activation=activation,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                name=params[3])
        if batch_norm:
            top = norm_layer(top, training, params[3]+'/batch_norm')
            top = tf.nn.relu(top, name=params[3]+'/relu')

        return top

    conv1 = conv_layer(inp, layer_params[0], training)
    pool1 = tf.layers.max_pooling2d(
        conv1, [2, 2], [2, 2], padding='same', name='pool1')

    conv2 = conv_layer(pool1, layer_params[1], training)
    pool2 = tf.layers.max_pooling2d(
        conv2, [2, 2], [2, 2], padding='same', name='pool2')

    conv3 = conv_layer(pool2, layer_params[2], training)
    conv4 = conv_layer(conv3, layer_params[3], training)
    pool3 = tf.layers.max_pooling2d(
        conv4, [2, 2], [2, 2], padding='same', name='pool3')

    conv5 = conv_layer(pool3, layer_params[4], training)
    conv6 = conv_layer(conv5, layer_params[5], training)
    conv7 = conv_layer(conv6, layer_params[6], training)

    return conv7

def build_model(inp, batch_size, num_rows, num_columns, dec_seq_len):
    #constants
    enc_lstm_dim = 256
    dec_lstm_dim = 512
    vocab_size = len(idx2vocab.keys())#40
    embedding_size = 20

    cnn = init_cnn(inp)#(16,40,240,1)==>(16, 1, 28, 512)
    #function for map to apply the rnn to each row
    def encoder(inp):#inp:shape=(16, 28, 512)
        enc_init_shape = [batch_size, enc_lstm_dim]#[16,256]
        with tf.variable_scope('encoder_rnn'):
            with tf.variable_scope('forward'):
                lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(enc_lstm_dim)
                init_fw = tf.nn.rnn_cell.LSTMStateTuple(\
                                    tf.get_variable("enc_fw_c", enc_init_shape),\
                                    tf.get_variable("enc_fw_h", enc_init_shape)
                                    )
            with tf.variable_scope('backward'):
                lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(enc_lstm_dim)
                init_bw = tf.nn.rnn_cell.LSTMStateTuple(\
                                    tf.get_variable("enc_bw_c", enc_init_shape),\
                                    tf.get_variable("enc_bw_h", enc_init_shape)
                                    )
            
            output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, \
                                                        lstm_cell_bw, \
                                                        inp, \
                                                        sequence_length = tf.fill([batch_size],\
                                                        tf.shape(inp)[1]),#(28,28...,28)\
                                                        initial_state_fw = init_fw, \
                                                        initial_state_bw = init_bw \
                                                        )#shape=((16, 28, 256),(16,28,256))

        return tf.concat(output,2)##shape=(16,28, 512)

    encoder = tf.make_template('fun', encoder)
    # shape is (batch size, rows, columns, features)
    # swap axes so rows are first. map splits tensor on first axis, so encoder will be applied to tensors
    # of shape (batch_size,time_steps,feat_size)
    rows_first = tf.transpose(cnn,[1,0,2,3])#shape=(1, 16, 28, 64)
    res = tf.map_fn(encoder, rows_first, dtype=tf.float32)#shape=(1, 16,28, 512)
    # encoder_output = tf.transpose(res,[1,0,2,3])#shape=(16, 1, 28, 512)
    encoder_output = tf.transpose(res,[1,2,0,3])#shape=(16,28,1,512)

    dec_lstm_cell = tf.nn.rnn_cell.LSTMCell(dec_lstm_dim)
    dec_init_shape = [batch_size, dec_lstm_dim]
    dec_init_state = tf.nn.rnn_cell.LSTMStateTuple( tf.truncated_normal(dec_init_shape),\
                                                    tf.truncated_normal(dec_init_shape) )#tuple:(c,h) c.shape=(16, 512)  h.shape=(16, 512)


    decoder_output = embedding_attention_decoder(dec_init_state,#[16, 512]第一个解码cell的state=[c,h]
                                                    tf.reshape(encoder_output,[batch_size, -1,2*enc_lstm_dim]),
                                                    #encoder输出reshape为 attention states作为attention模块的输入 shape=(16,28,512)
                                                    dec_lstm_cell,#512个lstm单元，作为解码层
                                                    vocab_size,#491
                                                    dec_seq_len,#8
                                                    batch_size,#16
                                                    embedding_size,#10
                                                    feed_previous=True)#dec_seq_len = num_words　= time_steps

    return (encoder_output, decoder_output)

def similarity(pred,true_labels):

    pred_str = []
    for i in pred:
        string = []
        for j in i:
            if j==1:
                string.append(idx2vocab[j])
                break
            string.append(idx2vocab[j])
        pred_str.append(''.join(string))

    true_str = []
    for i in true_labels:
        string = []
        for j in i:
            string.append(idx2vocab[j])
        true_str.append(''.join(string))

    s = []
    for i in range(len(pred_str)):
        d = Levenshtein.distance(pred_str[i],true_str[i])
        print("Pred[%d]:%s,True[%d]:%s \n"%(i,pred_str[i],i,true_str[i]))
        s_i = 1 - d/len(true_str[i])
        s.append(s_i)

    accu = np.mean(np.array(s))
    print("Accu:",accu)

    return accu

def main():

    global_steps = tf.Variable(0)
    batch_size = 16
    epochs = 100
    lr = 0.0001#注意:lr设小一点，否则loss几步之后不下降,无法收敛
    
    inp = tf.placeholder(shape=[None,None,None,1],dtype=tf.float32)
    num_rows = tf.placeholder(tf.int32)
    num_columns = tf.placeholder(tf.int32)
    num_words = tf.placeholder(tf.int32)
    true_labels = tf.placeholder(tf.int32,shape=[None,None])

    print ("Building Model")
    #测试代码段，查看中间变量以及调试attention
    # pdb.set_trace()
    # inp = tf.ones((16,40,240,1),tf.float32)
    # true_labels = tf.ones((16,8),tf.int32)
    # _, (output,state) = build_model(inp, batch_size, num_rows=40, num_columns=240, dec_seq_len=8)
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # output = sess.run(output)
    # pred = sess.run(tf.argmax( output, 2))
    # true = sess.run(true_labels)
    # sess.close()


    _, (output,state) = build_model(inp, batch_size, num_rows, num_columns, num_words)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_labels,logits=output))
    learning_rate = tf.train.exponential_decay(lr, global_steps, 1000, 0.9)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_steps)
    correct_prediction = tf.equal(tf.to_int32(tf.argmax( output, 2)), true_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #log variables
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('lr',learning_rate)
    merged_summary = tf.summary.merge_all()
    #load training data
    print ("Loading Data")
    train, val = load_data()
    train = batchify(train, batch_size)
    random.shuffle(train)
    val = batchify(val, batch_size)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 1#Session Config
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./log',sess.graph)
        print ("Training")
        for i in range(epochs):
            for j in range(len(train)):
                images, labels = train[j]#(16, 40, 240, 1),(16, 8)
                if labels.shape[0]!=16:
                    continue
                train_step.run(feed_dict={
                                    inp: images,\
                                    true_labels: labels,\
                                    num_rows: images.shape[1],\
                                    num_columns: images.shape[2],\
                                    num_words: labels.shape[1]})
                
                [pred,loss1,merged_summary1,steps,lr] = sess.run([tf.to_int32(tf.argmax( output, 2)),loss,merged_summary,global_steps,learning_rate],
                                                                feed_dict={
                                                                    inp: images,\
                                                                    true_labels: labels,\
                                                                    num_rows: images.shape[1],\
                                                                    num_columns: images.shape[2],\
                                                                    num_words: labels.shape[1]})

                writer.add_summary(merged_summary1,steps)
                print ("Epoch:%d,step:%d"%(i,j))
                print ("Cross_Entropy Loss:",str(loss1))
                print("global step:%d"%(steps))
                print("learning rate:%f"%(lr))
                accu = similarity(pred, labels)

                if steps%1000==0:
                    for k in range(len(val)):
                        images,labels = val[k]
                        if labels.shape[0] != 16:
                            continue
                        pred = sess.run(tf.to_int32(tf.argmax(output, 2)),
                                                    feed_dict={
                                                        inp: images,
                                                        num_rows: images.shape[1],
                                                        num_columns: images.shape[2],
                                                        num_words: labels.shape[1]})
                        print("=====Begin  valuating======")
                        accu = similarity(pred, labels)


            print ('Saving model')
            saver = tf.train.Saver()
            id = 'saved_models/model-'+time.strftime("%d-%m-%Y--%H-%M")
            saver.save(sess, id )

if __name__ == "__main__":
    main()
