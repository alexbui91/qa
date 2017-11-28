import tensorflow as tf


def get_max_pooling(_num_units, context, time_step_vector=None, prev_=None, reuse=None):
    """ sum over context to single vectzor"""
    context_ = tf.unstack(context, axis=1)
    if not time_step_vector is None or not prev_ is None:
        tmp = list()
        for c_ in context_:
            input_vector = list()
            input_vector.append(c_)
            if not time_step_vector is None:
                input_vector.append(time_step_vector)
            if not prev_ is None:
                input_vector.append(prev_)
            tmp.append(tf.concat(input_vector, axis=1))
        # B x L x D
        s_ = tf.transpose(tf.stack(tmp), perm=[1,0,2])
    else:
        s_ = context
    # # B x L x D
    s_ = tf.layers.dense(s_,
                        _num_units,
                        activation=tf.nn.tanh,
                        name="attention_question",
                        reuse=reuse)
    # To B x L x 1
    s_ = tf.layers.dense(s_,
                        1,
                        activation=None,
                        name="attention_question_softmax",
                        reuse=reuse)
    # To B x L
    # BxLxD => BxDxL x BxLx1 => BxD
    u_ = tf.squeeze(tf.matmul(tf.transpose(context, perm=[0,2,1]), s_))
    # return prediction and agg vector
    # expect a_: BxL, agg: BxD
    return tf.squeeze(s_), u_