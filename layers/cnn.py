import os, sys
proj_path = os.path.abspath('..')
sys.path.append(proj_path)

import tensorflow as tf
from layers.normalization import BNReLU, batch_norm, BatchNorm_Variant


def Conv2D(name, x, kernel_shape, stride, padding='SAME'):
    kernel = tf.get_variable(name, shape=kernel_shape, dtype=tf.float32)
    return tf.nn.conv2d(x, kernel, stride, padding)

def Conv1D(name, x, kernel_shape, stride, padding='SAME'):
    kernel = tf.get_variable(name, shape=kernel_shape, dtype=tf.float32)
    return tf.nn.conv1d(x, kernel, stride, padding)

def Gated_residual(name, param, kernel_shape, stride_shape, gated_channels, is_train=True):
    """ a residual block for GRN
    It is composed of three convolution layers, in which second layer contains gated mechanism
    Args:
        name: variable scope for this block
        param: input of this block, shape should be [B, T, F, C]
        kernel_shape: shape of convolution kernel, 2-rank of [T, F]
        stride_shape: shape of convolution stride, 2-rank of [T, F]
        gated_channels: gated layers may use differen channels, others' channel use input channel
        is_train: indicate it's training or test phase
    Returns:
        output tensor
    """
    input_shape = param.get_shape().as_list()
    in_channel = input_shape[-1]
    k_h, k_w = kernel_shape
    s_h, s_w = stride_shape
    with tf.variable_scope(name) as scope:
        kernel_shape1 = [k_h, k_w, in_channel, in_channel]
        stride = [1, s_h, s_w, 1]
        conv1 = Conv2D('conv1', param, kernel_shape1, stride)
        conv1 = BNReLU(conv1, [1, 1, 1, in_channel], [0, 1, 2], is_train=is_train, scope='bn1')
        # conv2 contains gated mechanism
        kernel_shape2 = [k_h, k_w, in_channel, gated_channels]
        conv2 = Conv2D('conv2', conv1, kernel_shape2, stride)  # linear activation
        gate = Conv2D('gate_conv', conv1, kernel_shape2, stride)
        gate = tf.nn.sigmoid(gate)  # gate use sigmoid activateion
        gated_conv2 = conv2 * gate
        kernel_shape3 = [k_h, k_w, gated_channels, in_channel]
        conv3 = Conv2D('conv3', gated_conv2, kernel_shape3, stride)
        conv3 = BNReLU(conv3, [1, 1, 1, in_channel], [0, 1, 2], is_train=is_train, scope='bn2')
        output = param + conv3
    return output


def Dilated_gated_res(name, param, k_size, r_dilated, neck_channels, is_train=True):
    """ a residual block for dilated gated residual block Wang DeLiang prompts
    This is for conv1d on timestep dim(stride is 1)
    Args:
        name: variable scope for this block
        param: input of this block, shape should be [B, T, C]
        k_size: kernel size in neck layer
        r_dilated: dilated coef for dilated conv1d in neck layer
        neck_channels: out_channels for layers except last layer
        is_train: indicate it's training or test phase
    Returns:
        output tensor
    """
    input_shape = param.get_shape().as_list()
    in_channels = input_shape[-1]
    with tf.variable_scope(name) as scope:
        res_input = BNReLU(param, [1, 1, in_channels], [0, 1], is_train=is_train, scope='bn1')
        #res_input = tf.nn.relu6(res_input)
        kernel_shape1 = [1, in_channels, neck_channels]
        stride_1d = 1
        conv1 = Conv1D('conv1', res_input, kernel_shape1, stride_1d)
        conv1 = BNReLU(conv1, [1, 1, neck_channels], [0, 1], is_train=is_train, scope='bn2')
        conv1 = tf.nn.relu6(conv1)
        # since api for dilated cnn can be used in conv2d, here need expand operation
        expand_conv1 = tf.expand_dims(conv1, axis=1)
        neck_kernel_shape = [1, k_size, neck_channels, neck_channels]
        stride_2d = [1, 1, 1, 1]
        gate_kernel = tf.get_variable('gate_conv', shape=neck_kernel_shape, dtype=tf.float32)
        conv2_kernel = tf.get_variable('conv2', shape=neck_kernel_shape, dtype=tf.float32)
        gate = tf.nn.conv2d(expand_conv1, gate_kernel, stride_2d,
                            padding='SAME', dilations=[1, 1, r_dilated, 1])
        gate = tf.nn.sigmoid(gate)
        conv2 = tf.nn.conv2d(expand_conv1, conv2_kernel, stride_2d,
                             padding='SAME', dilations=[1, 1, r_dilated, 1])
        # shape is [B, T, C]
        gated_conv2 = tf.squeeze(conv2 * gate, axis=1)
        kernel_shape2 = [1, neck_channels, in_channels]
        conv3 = Conv1D('conv3', gated_conv2, kernel_shape2, stride_1d)
        output = conv3 + param
    return output

