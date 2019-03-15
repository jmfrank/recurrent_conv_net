import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

class SpotCell( tf.nn.rnn_cell.RNNCell ):

	#Define properties. 
    def __init__(self, num_units, padding = 'same', layer_depth=64, conv_size=6, strides=1, n_classes=2):
        self._num_units = layer_depth
        self.padding=padding
        self.layer_depth = layer_depth
        self.conv_size=conv_size
        self.strides=strides
        self.n_classes = n_classes
    
    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__( self, inputs, state, scope=None):

        # First convolution
        with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
            
            outputs = tf.layers.conv2d(inputs,self.layer_depth, self.conv_size, padding=self.padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1',use_bias=True)
            outputs = tf.nn.relu(outputs)

        # Second convolution
        with tf.variable_scope("conv2", reuse=tf.AUTO_REUSE):

            outputs = tf.layers.conv2d(outputs,self.layer_depth, self.conv_size, padding=self.padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2',use_bias=True)
            outputs = tf.nn.relu(outputs)

        # Third convolution.
        with tf.variable_scope("conv3", reuse=tf.AUTO_REUSE):
            
            # Combine step
            outputs = tf.concat([state, outputs], 3)
            outputs = tf.layers.conv2d(outputs,self.layer_depth, self.conv_size, padding=self.padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3', use_bias=False)
            outputs = tf.nn.relu(outputs)

        # Reset state to outputs
        state = outputs

        # To class-based output.
        with tf.variable_scope("final", reuse=tf.AUTO_REUSE):
            outputs = tf.layers.conv2d(outputs,self.n_classes,self.conv_size, padding=self.padding, kernel_initializer=tf.contrib.layers.xavier_initializer(),name='final',use_bias=False)

        return outputs, state

def small_feature_net(nb_classes, inputs):

    padding = 'same'
    # input is in shape: ?,448,448,1
    
    #Start at 64-feature channels
    base=64
    
    #Convolution size
    conv_size=6
    strides=1
    
    #rename inputs.
    outputs = inputs
    
    this_name = 'conv1-1'
    
    # Conv block 1
    outputs = tf.layers.conv2d(outputs,base, conv_size, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=this_name,use_bias=True)
    outputs = tf.nn.relu(outputs)
    
    # Conv block 2
    this_name = 'conv1-2'
    outputs = tf.layers.conv2d(outputs,base, conv_size, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=this_name,use_bias=True)
    outputs = tf.nn.relu(outputs)
    
    # Up-sample(Conv_transpose)
    #outputs = tf.layers.conv2d_transpose(outputs, this_n, conv_size, strides=(2, 2), padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), use_bias=True)
    #outputs = tf.nn.relu(outputs)

    # Now output to the number of features.
    outputs = tf.layers.conv2d(outputs,nb_classes,conv_size,padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(),name='final',use_bias=False)

    return outputs


