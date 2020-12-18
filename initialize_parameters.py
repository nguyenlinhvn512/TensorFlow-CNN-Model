import numpy as np
import tensorflow as tf

def initialize_parameters():
    # """
    # Initializes weight parameters to build a neural network with tensorflow. The shapes are:
    #                     W1 : [4, 4, 3, 8]
    #                     W2 : [2, 2, 8, 16]
    # Note that we will hard code the shape values in the function to make the grading simpler.
    # Normally, functions should take values as inputs rather than hard coding.
    # Returns:
    # parameters -- a dictionary of tensors containing W1, W2
    # """

    # so that your "random" numbers match ours
    tf.compat.v1.set_random_seed(1)

    ### START CODE HERE ### (approx. 2 lines of code)
    W1 = tf.compat.v1.get_variable(
        "W1", [4, 4, 3, 8], initializer=tf.compat.v1.keras.initializers.glorot_normal(seed=0))
    W2 = tf.compat.v1.get_variable(
        "W2", [2, 2, 8, 16], initializer=tf.compat.v1.keras.initializers.glorot_normal(seed=0))
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters

