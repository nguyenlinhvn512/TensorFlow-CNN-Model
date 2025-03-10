import numpy as np
import tensorflow as tf

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    # """
    # Creates the placeholders for the tensorflow session.
    
    # Arguments:
    # n_H0 -- scalar, height of an input image
    # n_W0 -- scalar, width of an input image
    # n_C0 -- scalar, number of channels of the input
    # n_y -- scalar, number of classes
        
    # Returns:
    # X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    # Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    # """

    ### START CODE HERE ### (≈2 lines)
    X = tf.compat.v1.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf.compat.v1.placeholder(tf.float32, shape=(None, n_y))
    ### END CODE HERE ###

    return X, Y

