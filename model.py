#-*-coding:utf8-*-

__author="buyizhiyou"
__date="2018-10-30"


import sys,os
import numpy as np 
from vgg16 import vgg16 
import tensorflow as tf 
from config import config


'''
build vgg16+seq2seq(+attention) model
'''

# tf.enable_eager_execution()

def seq2seq(inp,output):
    vocab_size = config.vocab_size#70
    embed_dim = config.embed_dim#100
    num_units = config.num_units#256
    output_max_length = config.output_max_length#30
    batch_size = config.batch_size#8

    start_tokens = tf.zeros([batch_size], dtype=tf.int32)
    train_output = tf.concat([tf.expand_dims(start_tokens, 1), output], 1)#add START
    output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output, 3)), 1)-1#delete PADDING(3),END(1)

    output_embed = tf.contrib.layers.embed_sequence(
        train_output, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed')
    with tf.variable_scope('embed',reuse=tf.AUTO_REUSE):
        embeddings = tf.get_variable('embeddings')
    def encode(inp):#inp:shape=(4,?, 512)
        enc_init_shape = [batch_size, num_units]
        with tf.variable_scope('encode'):
            with tf.variable_scope('forward'):
                lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units)
                init_fw = tf.nn.rnn_cell.LSTMStateTuple(\
                                    tf.get_variable("enc_fw_c", enc_init_shape),\
                                    tf.get_variable("enc_fw_h", enc_init_shape)
                                    )
            with tf.variable_scope('backward'):
                lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units)
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

        return tf.concat(output,2)#shape=(4,?, 512)

    def decode(helper,reuse=None):
        with tf.variable_scope('decode', reuse=reuse):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units, memory=encoder_outputs)
            cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism, attention_layer_size=num_units / 2)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, vocab_size, reuse=reuse
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell, helper=helper,
                initial_state=out_cell.zero_state(
                    dtype=tf.float32, batch_size=batch_size))
                #initial_state=encoder_final_state)
            outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True,maximum_iterations=output_max_length)
            return outputs[0]

    with tf.variable_scope('trained'):
        encoder_outputs = encode(inp)
        train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, output_lengths)
        # train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
        #     output_embed, output_lengths, embeddings, 0.3
        # )
        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embeddings, start_tokens=tf.to_int32(start_tokens), end_token=1)
        train_outputs = decode(train_helper)
        pred_outputs = decode(pred_helper,reuse=True)
        # import pdb; pdb.set_trace()
        # tf.add_to_collection('pred_outputs',pred_outputs)
        tf.identity(train_outputs.sample_id[0], name='train_outputs_sq')
        tf.identity(pred_outputs.sample_id[0],name='pred_outputs_sq')
        weights = tf.to_float(tf.not_equal(train_output[:, :-1], 1))
        
    return train_outputs,pred_outputs,weights


if __name__=="__main__":

    import pdb; pdb.set_trace()
    global_steps = tf.Variable(0)
    lr = 0.00001
    batch_size = config.batch_size

    img =  tf.constant(np.random.rand(8,30,200,1),dtype=tf.float32)
    true_labels =  tf.constant(np.random.randint(30,size=(8,5)),dtype=tf.int32)

    conv = vgg16(img)
    encoder_inputs = conv[:,0,:,:]
    train_outputs,pred_outputs,weights = seq2seq(encoder_inputs,true_labels)

    loss = tf.contrib.seq2seq.sequence_loss(train_outputs.rnn_output, true_labels, weights=weights)
    learning_rate = tf.train.exponential_decay(lr, global_steps, 1000, 0.9)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_steps)



