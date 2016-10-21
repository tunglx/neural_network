import random
import math
from active_function import ReLU_function
from collections import namedtuple
import numpy as np

class Neuron:
    def __init__(self):
        self.bias = 0
        self.weights = []
        self.output = 0
        self.delta = 0

    def derivative_output(self):
        return self.output * (1 - self.output)

    def derivative_respect_to_target(self, target_output):
        return -(target_output - self.output)

    def error(self, target_output):
        return sum_squared_error(target_output, self.output)

    def calc_delta(self, target_output):
        return self.derivative_respect_to_target(target_output) * self.derivative_output()

class Layer:
    def __init__(self, num_neuron, num_weight):

        self.neurons = []

        for i in range(num_neuron):
            neuron = Neuron()
            neuron.bias = random.random()
            for j in range(num_weight):
                neuron.weights.append(random.random())

            self.neurons.append(neuron)

    # Calculate the value of neuron with inputs value and pass through active function
    def feed_forward(self, inputs):
        outputs = []
        inputs_numpy = np.array(inputs)

        for i in range(len(self.neurons)):
            weights_numpy = np.array(self.neurons[i].weights)
            self.neurons[i].output = ReLU_function(np.inner(inputs_numpy, weights_numpy) + self.neurons[i].bias)
            outputs.append(self.neurons[i].output)
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs