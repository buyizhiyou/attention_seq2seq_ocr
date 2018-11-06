import tensorflow as tf

tf.enable_eager_execution()

batch_size = 4 
input = tf.random_normal(shape=[3, batch_size, 6], dtype=tf.float32)
cell = tf.nn.rnn_cell.BasicLSTMCell(10, forget_bias=1.0, state_is_tuple=True)
init_state = cell.zero_state(batch_size, dtype=tf.float32)
seq_length = tf.constant([2,3,2,3],dtype=tf.int32)
import pdb; pdb.set_trace()
output, final_state = tf.nn.dynamic_rnn(cell, input, initial_state=init_state,sequence_length=seq_length,time_major=True) #time_major如果是True，就表示RNN的steps用第一个维度表示，建议用这个，运行速度快一点。
#如果是False，那么输入的第二个维度就是steps。
#如果是True，output的维度是[steps, batch_size, depth]，反之就是[batch_size, max_time, depth]。就是和输入是一样的
#final_state就是整个LSTM输出的最终的状态，包含c和h。c和h的维度都是[batch_size， n_hidden]
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run([output,final_state]))