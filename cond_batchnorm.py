import numpy as np
import tensorflow as tf


def Batchnorm(axes, inputs, name=None,is_training=None, stats_iter=None, update_moving_stats=True, fused=True, labels=None,
              n_labels=None):
    """conditional batchnorm (dumoulin et al 2016) for BCHW conv filtermaps"""

    if axes == [0, 1, 2]:
        mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
        shape = mean.get_shape().as_list()  # shape is [1,1,1,n]
        with tf.variable_scope("condBN", reuse=tf.AUTO_REUSE):
            offset_m = tf.get_variable(".offset", shape=[n_labels, shape[3]],initializer=tf.zeros_initializer(), dtype='float32')
            scale_m = tf.get_variable(".scale",  shape=[n_labels, shape[3]],initializer=tf.ones_initializer(), dtype='float32')
        offset = tf.nn.embedding_lookup(offset_m, labels)
        scale = tf.nn.embedding_lookup(scale_m, labels)
        print(inputs.shape, mean.shape, var.shape, offset.shape, scale.shape)
        result = tf.nn.batch_normalization(inputs, mean, var, offset[:, None, None,:], scale[:,None, None, :], 1e-5)
    elif axes==[0]:
        mean, var = tf.nn.moments(inputs, axes, keep_dims=False)

        shape = mean.get_shape().as_list()  # shape is [1,n]
        with tf.variable_scope("condBN", reuse=tf.AUTO_REUSE):
            offset_m = tf.get_variable(".offset", shape=[n_labels, shape[0]],initializer=tf.zeros_initializer(), dtype='float32')
            scale_m = tf.get_variable(".scale",  shape=[n_labels, shape[0]],initializer=tf.ones_initializer(), dtype='float32')
        offset = tf.nn.embedding_lookup(offset_m, labels)
        scale = tf.nn.embedding_lookup(scale_m, labels)
        print(inputs.shape,mean.shape,var.shape,offset.shape,scale.shape)
        result = tf.nn.batch_normalization(inputs, mean, var, offset[:,None], scale[:,None], 1e-5) #adds to (?,?,4096) for no reason
    else:
        raise Exception('unsupported')
    print(result.shape)
    return result
