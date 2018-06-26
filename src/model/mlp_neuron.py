import random
import numpy as np
from util.activation_functions import Activation


class MLPNeuron(object):
    def __init__(self, num_weights, activation="sigmoid", bias=True):
        # Randomly initialize weights
        self.weights = [random.uniform(0, 1) for i in range(num_weights)]
        self.bias = random.uniform(0, 1) if bias else 0
        self.activation = activation

    def compute_output(self, input):
        if len(input) != len(self.weights):
            raise ValueError("MLPNeuron: Bad input dimensions: "
                             "Got vector of length {}, expected {}".format(len(input), len(self.weights)))
        weighted_sum = np.dot(input, self.weights) + self.bias
        return Activation.getActivation(self.activation)(weighted_sum)