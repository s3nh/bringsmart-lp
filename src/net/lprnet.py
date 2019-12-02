import tensorflow as tf
import numpy as np 
import time 
import cv2 
import os 
import random 
import yaml 

CONFIG_PATH = 'config/config.yaml'

def load_config(config_path = CONFIG_PATH):
    assert os.path.exists(CONFIG_PATH)
    with open(config_path, 'r') as confile:
        config = yaml.safe_load(confile)
    return config

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
    conv_biases = tf.Variable(tf.zeros([omap], dtype = np.tf.float32))
    out = tf.nn.conv2d(x, weights, strides = stride, padding = pad)
    relu = tf.nn.bias_add(out, conv_biases)
    return relu

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
 
def concatenate_block(x, ksize = [1, 4, 1, 1], strides = [1, 4, 1, 1]):
    x = tf.nn.avg_pool(x,
    ksize = ksize, 
    strides = strides, 
    padding = 'SAME' 
  )
    cx = tf.reduce_mean(tf.square(x))
    x = tf.div(x, cx)
    return x


def train_model(num_channels, label_len, b, img_size):
    """
    Define forward pass
    and backward pass
    """
    # Define inputs 
    inputs = tf.placeholder(tf.float32, shape=(b, img_size[0], img_size[1], num_channels))
    #ctc loss
    targets = tf.sparse_placeholder(tf.int32)
    seq_len = tf.placeholder(tf.int32, [None])
    x = inputs
    x = conv(x, num_channels, 64, ksize=[3,3])
    # Batch Norm
    x = tf.layers.batch_normalization(x)
    # ReLU activation
    x = tf.nn.relu(x)
    # Max Pooling 
    x = tf.nn.max_pool(x , ksize =[1,3,3,1], strides = [1,1,1,1], padding='SAME')
    x = small_basic_block(x, 64, 64)
    # Divide into another layer
    x2 = x
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x, ksize = [1,3,3,1], 
        strides = [1,2,1,1], 
        padding = 'SAME')
    x = small_basic_block(x, 64, 256)
    x = tf.layers.batch_normalization(x)
    tf.nn.relu(x)
    # Divide into another layer
    x3 = x
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x, 
    ksize = [1,3,3,1], 
    strides = [1,2,1,1], 
    padding='SAME')
    x = tf.layers.dropout(x)
    x = conv(x, 256, 256, ksize=[4,1])
    x = tf.layers.dropout(x)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu()


    x = conv(x, 256, NUM_CHARS + 1, ksize=[1, 13], pad = 'SAME')
    x = tf.nn.relu(x)
    cx = tf.reduce_mean(tf.square(x))
    x = tf.div(x, cx) 
    """
    x1 = tf.nn.avg_pool(inputs, 
    ksize = [1, 4, 1, 1], 
    strides = [1, 4, 1, 1],
    padding = 'SAME')
    cx1 = tf.reduce_mean(tf.square(x1))
    x1 = tf.div(x1, cx1)
    x2 = tf.nn.avg_pool(inputs, 
    ksize = [1, 4, 1, 1], 
    strides = [1, 4, 1, 1], 
    padding =  'SAME')
    cx2 =  tf.reduce_mean(tf.square(x2))
    x2 = tf.div(x2, cx2)
    """
    x1 = concatenate_block(x1)
    x2 = concatenate_block(x2)
    x3 = concatenate_block(x3, ksize = [1, 2, 1, 1], strides = [1,2,1,1])
    # Layers concatenation 
    x = tf.concat([x1, x2, x3], 3)
    x = conv(x, x.get_shape().as_list()[3], NUM_CHARS+1, ksize=(1,1))
    logits = tf.reduce_mean(x, axis=2)
    return logits, inputs, targets, seq_len

def main():
    confile = load_config()
    print(confile)

if __name__ == "__main__":
    main()