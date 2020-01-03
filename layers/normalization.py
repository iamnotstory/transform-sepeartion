import os, sys
proj_path = os.path.abspath('..')
sys.path.append(proj_path)

import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average

def zeros_init():
    return tf.zeros_initializer()

def ones_init():
    return tf.ones_initializer()

def batch_norm(param, masked_param, shape, axes, offset=True, scale=True, eps=1e-8,
               decay=0.99, dtype=tf.float32, scope=None, is_train=True):
    with tf.variable_scope(scope or "BatchNorm"):
        name = param.op.name.split('/')[-1]
        running_mean = tf.get_variable('{}_running_mean'.format(name),
                                       shape=shape, initializer=zeros_init(),
                                       trainable=False, dtype=dtype)
        running_var = tf.get_variable('{}_running_var'.format(name),
                                      shape=shape, initializer=ones_init(),
                                      trainable=False, dtype=dtype)
        offset_var = None
        if offset:
            offset_var = tf.get_variable('{}_offset'.format(name),
                                         shape=shape, initializer=zeros_init(),
                                         dtype=dtype)
        scale_var = None
        if scale:
            scale_var = tf.get_variable('{}_scale'.format(name),
                                        shape=shape, initializer=ones_init(),
                                        dtype=dtype)
        def batch_statistics():
            mean = tf.reduce_mean(masked_param, axes, keep_dims=True)
            var = tf.reduce_mean(tf.square(masked_param - mean), axes, keep_dims=True)
            update_running_mean = assign_moving_average(
                running_mean, mean, decay, zero_debias=False)
            update_running_var = assign_moving_average(
                running_var, var, decay, zero_debias=False)
            with tf.control_dependencies([update_running_mean, update_running_var]):
                normed_param = tf.nn.batch_normalization(
                    param, mean, var, offset_var, scale_var, eps,
                    '{}_bn'.format(name))
            return normed_param
        def population_statistics():
            normed_param = tf.nn.batch_normalization(
                param, running_mean, running_var, offset_var, scale_var,
                eps, '{}_bn'.format(name))
            return normed_param
        normed_param = tf.cond(is_train, batch_statistics, population_statistics)
        return normed_param, running_mean, running_var


def BatchNorm_Variant(param, shape, axes, offset=True, scale=True, eps=1e-8,
                      decay=0.99, dtype=tf.float32, scope=None, is_train=True):
    """
    variant of origin batch_norm where running_mean and running_var used both in 
    training phase and test phase
    """
    with tf.variable_scope(scope or "BatchNorm"):
        running_mean = tf.get_variable('running_mean', shape=shape, initializer=zeros_init(),
                                       trainable=False, dtype=dtype)
        running_var = tf.get_variable('running_var', shape=shape, initializer=ones_init(),
                                      trainable=False, dtype=dtype)
        offset_var = None
        if offset:
            offset_var = tf.get_variable('offset', shape=shape, dtype=dtype,
                                         initializer=zeros_init())
        scale_var = None
        if scale:
            scale_var = tf.get_variable('scale', shape=shape, dtype=dtype,
                                        initializer=ones_init())
        def train_phase():
            mean = tf.reduce_mean(param, axes, keep_dims=True)
            var = tf.reduce_mean(tf.square(param - mean), axes, keep_dims=True)
            update_running_mean = assign_moving_average(
                running_mean, mean, decay, zero_debias=False)
            update_running_var = assign_moving_average(
                running_var, var, decay, zero_debias=False)
            with tf.control_dependencies([update_running_mean, update_running_var]):
                normed_param = tf.nn.batch_normalization(
                    param, running_mean, running_var, offset_var, scale_var, eps, name='bn')
            return normed_param
        def test_phase():
            normed_param = tf.nn.batch_normalization(
                param, running_mean, running_var, offset_var, scale_var, eps, name='bn')
            return normed_param
        normed_param = tf.cond(is_train, train_phase, test_phase)
        return normed_param, running_mean, running_var


def layer_normalization(param, dims, axis=-1, offset=True, scale=True,
                        name=None, eps=1e-8, dtype=tf.float32, scope=None):
    with tf.variable_scope(scope or "LayerNorm"):
        if name is None:
            name = param.op.name.split('/')[-1]
        offset_var = 0
        if offset:
            offset_var = tf.get_variable(name+'_offset', shape=[dims],
                                         initializer=tf.zeros_initializer(),
                                         dtype=dtype)
        scale_var = 1
        if scale:
            scale_var = tf.get_variable(name+'_scale', shape=[dims],
                                        initializer=tf.ones_initializer(),
                                        dtype=dtype)
        mean = tf.reduce_mean(param, axis=axis, keep_dims=True)
        inverse_stddev = tf.rsqrt(tf.reduce_mean(
            tf.square(param - mean), axis=axis, keep_dims=True) + eps)
        normed = (param - mean) * inverse_stddev
        return normed * scale_var + offset_var


def BNReLU(param, shape, axes, offset=True, scale=True, eps=1e-8,
           decay=0.99, dtype=tf.float32, scope=None, is_train=True):
    """
    a shorthand of BatchNormalization + ReLU for 2d data
    """
    param, _, _  = batch_norm(param, param, shape, axes, offset, scale,
                              eps, decay, dtype, scope, is_train)
    param = tf.nn.relu6(param)
    return param
