import tensorflow as tf
import numpy as np 
import time 
import cv2 
import os 
import random 


def conv(x, imap, omap, ksize, stride = [1, 1, 1, 1], pad = 'SAME'):
    """
    Create conv input for 
    small basic block creation 
    Return:
        feature maps after ReLU activation 
        with conv_biases added 
    """
    weights = tf.Variable(tf.truncated_normal([ksize[0], ksize[1], imap, omap], 
    stddev= 0.1, 
    seed=None, 
    dtype = tf.float32))
    conv_biases = tf.Variable(tf.zeros([om], dtype = np.tf.float32))
    out = tf.nn.conv2d(x, weights, strides = stride, padding = pad)
    relu = tf.nn.bias_add(out, conv_biases)
    return relu
    # Add biases 


#LPRNEt: 

#https://arxiv.org/pdf/1806.10447.pdf

def small_basic_block(x, imap, omap):
    """
    small basic block architecture, 
    which contain 
    input CxHxW 
    Conv Cout/4 1x1 stride 1 
    Conv Cout 4 3x1 strideh=1. padh = 1
    Conv Cout 4 1x3 stridew=1, padw = 1
    COnv Cout 1x1 stride 1
    Output CoutxHxW feature map 
    """
    x = conv(x, im, int(om/4), ksize=[1,1])
    x = tf.nn.relu(x)
    #FIXME
    x = conv(x, int(om/4), int(om/4), ksize =[3,1], pad='SAME')
    x = tf.nn.relu(x)
    x = conv(x, int(om/4), int(om/4), ksize = [1,3], pad='SAME')
    x = tf.nn.relu(x)
    x = conv(x, int(om/4), om, ksize=[1,1])
    return x