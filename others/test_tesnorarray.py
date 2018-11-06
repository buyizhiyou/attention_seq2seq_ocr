import tensorflow as tf 
import numpy as np 

B=3
D=4
T=5

tf.reset_default_graph()
xs=tf.placeholder(shape=[T,B,D],dtype=tf.float32)

with tf.variable_scope('rnn'):
    GRUcell = tf.nn.rnn_cell.GRUCell(num_units=D)
    cell = tf.nn.rnn_cell.MultiRNNCell([GRUcell])

    output_ta = tf.TensorArray(size=T,dtype=tf.float32)
    input_ta = tf.TensorArray(size=T,dtype=tf.float32)
    input_ta = input_ta.unstack(xs)

    def body(time,output_ta_t,state):
        xt = input_ta.read(time)
        new_output,new_state = cell(xt,state)
        output_ta_t = output_ta_t.write(time, new_output)
        return (time+1,output_ta_t,new_state)

    def condition(time,output,state):
        return time<T
    
    time=0
    state=cell.zero_state(B,tf.float32)
    time_final,output_ta_final,state_final=tf.while_loop(cond=condition,body=body,loop_vars=(time,output_ta,state))
    output_final = output_ta_final.stack()

x=np.random.randn(T,B,D)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_final_,state_final_=sess.run([output_final,state_final],feed_dict={xs:x})


'''

import tensorflow as tf
tf.enable_eager_execution()

def condition(time,max_time, output_ta_l):
    return tf.less(time, max_time)

def body(time,max_time, output_ta_l):
    output_ta_l = output_ta_l.write(time, [2.4, 3.5])
    return time + 1, max_time,output_ta_l

max_time=tf.constant(3)
time = tf.constant(0)
output_ta = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
result = tf.while_loop(condition, body, loop_vars=[time,max_time,output_ta])
last_time,max_time, last_out = result
final_out = last_out.stack()


print(last_time)
print(final_out)

'''
'''
ta.stack(name=None) 将TensorArray中元素叠起来当做一个Tensor输出
ta.unstack(value, name=None) 可以看做是stack的反操作，输入Tensor，输出一个新的TensorArray对象
ta.write(index, value, name=None) 指定index位置写入Tensor
ta.read(index, name=None) 读取指定index位置的Tensor
作者：加勒比海鲜 
原文：https://blog.csdn.net/guolindonggld/article/details/79256018 
'''
