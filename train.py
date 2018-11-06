#-*-coding:utf8-*-

__author="buyizhiyou"
__date="2018-10-30"

import sys,os
import numpy  as np 
from config import config
from model import seq2seq
from vgg16 import vgg16 
from preprocess import load_data,batchify

import tensorflow as tf 
import Levenshtein
from editdistance import edit

os.environ['VISIBLE_CUDA_DEVICES']='0'

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

os.environ['CUDA_VISIBLE_DEVICES']='1'

def accu(trues,preds):
    '''
    trues:[batch_size,length]
    preds:[batch_size,length2]
    '''

    dist = lambda true,pred:Levenshtein.distance(true,pred)/max(len(true),len(pred))
    batch_size = trues.shape[0]
    distances = 0
    for i in range(batch_size):
        true = trues[i]
        pred = preds[i]
        true = [idx2vocab[e] for e in true]
        pred = [idx2vocab[e] for e in pred]
        true = ''.join(true)
        pred = ''.join(pred)
        distances +=dist(true,pred)
    
    return 1-distances/batch_size

def train_seq2seq():
    #config
    batch_size = config.batch_size
    lr = config.learning_rate
    epoches = config.epoches
    log_path = config.log_path
    checkpoint_path = config.checkpoint_path
    pretrained_model = config.pretrained_model

    img = tf.placeholder(shape=[None,None,None,3],dtype=tf.float32)
    true_labels = tf.placeholder(tf.int32,shape=[None,None])

    #build network and train_step
    conv = vgg16(img)
    encoder_inputs = conv[:,0,:,:]
    train_outputs,pred_outputs,weights = seq2seq(encoder_inputs,true_labels)
    global_steps = tf.Variable(0)

    loss = tf.contrib.seq2seq.sequence_loss(train_outputs.rnn_output, true_labels, weights=weights)
    learning_rate = tf.train.exponential_decay(lr, global_steps, 1000, 0.9)
    # var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trained')#only train seq2seq part
    # train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_steps,var_list=var_list)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_steps)

    tf.summary.scalar('loss',loss)
    sum_ops = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_path,graph=tf.get_default_graph())

    with tf.Session() as sess:
        #initializer variables
        sess.run(tf.global_variables_initializer())

        #load pretrained model from .npy file
        # vars = tf.trainable_variables()
        # vars = [var for var in vars if 'vgg16' in var.name]
        def init_vgg16():
            print("Init vgg16 from Imagenet_vgg16.npy file!")
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='vgg16')
            data_dict = np.load(pretrained_model,encoding='latin1').item()
            keys = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3']
            for i in range(0,len(vars),2):
                var1 = vars[i]
                var2 = vars[i+1]
                key = keys[i//2]
                sess.run(var1.assign(data_dict[key]['weights']))
                sess.run(var2.assign(data_dict[key]['biases']))

        model_file = tf.train.latest_checkpoint(checkpoint_path)#restore model
        saver = tf.train.Saver()
        if model_file:
            print("model restored from {}".format(model_file))
            saver.restore(sess,model_file)
        else:
            init_vgg16()

        for epoch in range(epoches):
            data = load_data('data/train.txt')
            batch = batchify(data,batch_size)
            step = 0  
            while True:
                try:
                    batch1 = next(batch)
                    train_img = batch1[0]
                    train_true_labels = batch1[1]
                    # import pdb; pdb.set_trace()
                    global_steps1 = sess.run(global_steps)
                    loss1,_ = sess.run([loss,train_step],feed_dict={img:train_img,true_labels:train_true_labels})
                    summary = sess.run(sum_ops,feed_dict={img:train_img,true_labels:train_true_labels})
                    print("epoch:{},step:{},loss:{}".format(epoch,step,loss1))
                    summary_writer.add_summary(summary,global_step=global_steps1)
                    step+=1

                    if step%1000==0:
                        print("#############Val###############")
                        Accu = 0
                        j=0
                        data_val = load_data('data/val.txt')
                        batch_val = batchify(data_val,batch_size)
                        while True:
                            try:
                                batch2 = next(batch_val)
                                val_img=batch2[0]
                                # labels =np.expand_dims(list(range(batch2[1].shape[1])),axis=0)#according to img's length ,we can get the text length
                                # val_true_labels = np.repeat(labels,batch_size,axis=0)
                                val_true_labels = batch2[1]
                                pred_outputs1 = sess.run([pred_outputs],feed_dict={img:val_img,true_labels:val_true_labels})
                                y_pred = pred_outputs1[0].sample_id
                                accu1 = accu(val_true_labels,y_pred)
                                Accu+=accu1
                                j+=1
                                print("   Accu:{}".format(accu1))

                            except StopIteration:
                                print("Mean Accu:{}".format(Accu/j))
                                break
                            
                except StopIteration:
                    break


            #save model checkpoint
            print("model saved!")
            saver.save(sess,checkpoint_path+'/models.ckpt',global_step=epoch)

if __name__ == "__main__":
    import pdb; pdb.set_trace()
    train_seq2seq()