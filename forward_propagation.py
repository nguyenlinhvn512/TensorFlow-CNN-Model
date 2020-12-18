import numpy as np
import tensorflow as tf

def forward_propagation(X, parameters):
    # """
    # Implements the forward propagation for the model:
    # CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    # Note that for simplicity and grading purposes, we'll hard-code some values
    # such as the stride and kernel (filter) sizes. 
    # Normally, functions should take these values as function parameters.
    
    # Arguments:
    # X -- input dataset placeholder, of shape (input size, number of examples)
    # parameters -- python dictionary containing your parameters "W1", "W2"
    #               the shapes are given in initialize_parameters

    # Returns:
    # Z3 -- the output of the last LINEAR unit
    # """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']

    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[
                        1, 8, 8, 1], padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[
                        1, 4, 4, 1], padding='SAME')
    # FLATTEN
    F = tf.compat.v1.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z3 = tf.compat.v1.layers.dense(F, 6)
    ### END CODE HERE ###

    return Z3
