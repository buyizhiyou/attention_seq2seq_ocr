"""
This is a modified version of tensorflow.nn.seq2seq's attention_decoder and 
embedding_attention_decoder which allow dynamic sequence lengths and implement
the specific calculations dictated in the im2markup paper. It will probably be
rewritten as tensorflow gains dynamic variants of its seq2seq models, such as 
the recently added dynamic_rnn_decoder.
"""
import numpy as np
import tensorflow as tf
import pdb
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

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell_impl._linear    # pylint: disable=protected-access

def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """

  def loop_function(prev, _):
      #prev:16x40
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])#prev:16x40
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)#从embedding矩阵查找,embedding:40x20,prev_symbol:16
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev

  return loop_function

def attention_decoder(initial_state,#(c,h)  c.shape/h.shape:(16, 512)
                      attention_states,#shape=(16, 28, 512)
                      cell,
                      vocab_size,#40
                      time_steps,#num_words,8
                      batch_size,#16
                      output_size=None,#512
                      loop_function=None,
                      dtype=None,
                      scope=None):
    if attention_states.get_shape()[2].value is None:#tf 张量　get_shape()方法获取size
        raise ValueError("Shape[2] of attention_states must be known: %s"
                                         % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size#512

    with variable_scope.variable_scope(scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype

        attn_length = attention_states.get_shape()[1].value #28
        if attn_length is None:
            attn_length = shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value#512

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])#shape=(16, 28, 1, 512) 
        attention_vec_size = attn_size    # Size of query vectors for attention.   512
        k = variable_scope.get_variable("AttnW",[1, 1, attn_size, attention_vec_size])#shape=(1,1,512,512)
        hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")#(16 ,28, 1, 512) w_1*h_j
        v = variable_scope.get_variable("AttnV", [attention_vec_size])#shape:(512,)


        def attention(query):
            #LSTMStateTuple(c= shape=(16, 512) dtype=float32>, h=< shape=16, 512) dtype=float32>)
            """Put attention masks on hidden using hidden_features and query."""
            if nest.is_sequence(query):    # If the query is a tuple, flatten it.
                query_list = nest.flatten(query) #[c,h]，第一个随即初始化，以后调用之前计算的
                for q in query_list:    # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list,1)# shape=(16, 1024)
            with variable_scope.variable_scope("Attention_0"):
                y = linear(query, attention_vec_size, True)# shape=(16, 512) w_2*s_t
                y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size]) # shape=(16, 1, 1, 512)
                s = math_ops.reduce_sum( v * math_ops.tanh(hidden_features+y), [2, 3])#!!!!!!!!!!!公式(3)用一个小神经网络拟合attention中权重系数shape=(16, 28)
                a = nn_ops.softmax(s)#  公式(2)shape=(16, 28),softmax归一化系数
                # Now calculate the attention-weighted vector d.
                d = math_ops.reduce_sum(
                        array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,#！！！！！！！公式(1)attention计算，加权求和
                        [1, 2])#shape=(16, 512) 
                ds = array_ops.reshape(d, [-1, attn_size])#shape=(16, 512) #!!!!!!!!!!!!以上是attention　model中三个关键公式的实现
            return ds
        batch_attn_size = array_ops.stack([batch_size, attn_size]) #(16,512)
        attn = array_ops.zeros(batch_attn_size, dtype=dtype)#shape=(16, 512)
        attn.set_shape([None, attn_size])#(16,512)

        def cond(time_step, prev_softmax_input, state_c, state_h, outputs2):
            return time_step < time_steps

        def body(time_step, prev_softmax_input, state_c, state_h, outputs2):
            #outputs:shape=(16, ?, 40) prev_softmax_input=init_word:shape=(16, 40)
            state = tf.nn.rnn_cell.LSTMStateTuple(state_c,state_h)#第一次随机初始状态，之后调用之前的
            attns = attention(state)#shape=(16, 512) attenion模块的输出,C_i
            with variable_scope.variable_scope("loop_function", reuse=True):
                inp = loop_function(prev_softmax_input, time_step)#shape=(16,10)

            x = tf.concat((inp, attns),1)
            cell_output, state = cell(x, state)#decoder层512个lstm单元 cell_output:shape=(16, 512) state:shape=(16, 512)
            # Run the attention mechanism.
            with variable_scope.variable_scope("AttnOutputProjection"):
                softmax_input = linear(cell_output, vocab_size, False)

            new_outputs = tf.concat([outputs2,tf.expand_dims(softmax_input,1)],1)#shape=(16, ?, 40)[,...y_t-1,y_t,...]

            return (time_step + tf.constant(1, dtype=tf.int32),\
                            softmax_input, state.c, state.h, new_outputs)#既是输出，又是下一轮的输入

        time_step = tf.constant(0, dtype=tf.int32)
        shape_invariants = [time_step.get_shape(),\
                            tf.TensorShape([batch_size, vocab_size]),\
                            tf.TensorShape([batch_size,512]),\
                            tf.TensorShape([batch_size,512]),\
                            tf.TensorShape([batch_size, None, vocab_size])]

       # START keyword is random
        init_word = np.random.rand(batch_size, vocab_size)#shape=(16,40)

        loop_vars = [time_step,\
                     tf.constant(init_word, dtype=tf.float32),\
                     initial_state.c,initial_state.h,\
                     tf.zeros([batch_size,1,vocab_size])] # we just need to feed an empty matrix
                                                          # to start off the while loop since you can
                                                          # only concat matrices that agree on all but
                                                          # one dimension. Below, we remove that initial
                                                          # filler index

        outputs = tf.while_loop(cond, body, loop_vars, shape_invariants)##shape=(16, ?, 40)
        '''
        loop_vars = [...]
        while cond(*loop_vars):
            loop_vars = body(*loop_vars)   
        '''

    return outputs[-1][:,1:], tf.nn.rnn_cell.LSTMStateTuple(outputs[-3],outputs[-2])

def embedding_attention_decoder(initial_state,#(c,h)
                                attention_states,# shape=(16, 28, 512)
                                cell,
                                num_symbols,#40
                                time_steps,
                                batch_size,#16
                                embedding_size,#20
                                output_size=None,#512
                                output_projection=None,
                                feed_previous=False,#True
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None):
    if output_size is None:
        output_size = cell.output_size#512
    if output_projection is not None:
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    with variable_scope.variable_scope(scope or "embedding_attention_decoder", dtype=dtype) as scope:
        embedding = variable_scope.get_variable("embedding",[num_symbols, embedding_size])#shape=(40,20)
        loop_function = _extract_argmax_and_embed(embedding,
                          output_projection,update_embedding_for_previous) if feed_previous else None
                        #(16,40)==>(16,20)找argmax,然后embedding
        return attention_decoder(
                initial_state,
                attention_states,
                cell,
                num_symbols,#40
                time_steps,#8
                batch_size,
                output_size=output_size,#512
                loop_function=loop_function)


