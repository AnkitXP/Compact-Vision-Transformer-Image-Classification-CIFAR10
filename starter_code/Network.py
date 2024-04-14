### YOUR CODE HERE
import torch

"""This script defines the network.
"""

class MyNetwork(object):

    def __init__(self, configs):
        self.configs = configs

    def __call__(self, inputs, training):
    	'''
    	Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Used by operations that work differently
                in training and testing phases such as batch normalization.
        Return:
            The output Tensor of the network.
    	'''
        return self.build_network(inputs, training)

    def build_network(self, inputs, training):
        return inputs


### END CODE HERE