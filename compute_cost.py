import numpy as np
import tensorflow as tf

def compute_cost(Z3, Y):
    # """
    # Computes the cost
    
    # Arguments:
    # Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (number of examples, 6)
    # Y -- "true" labels vector placeholder, same shape as Z3
    
    # Returns:
    # cost - Tensor of the cost function
    # """

    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    ### END CODE HERE ###

    return cost

