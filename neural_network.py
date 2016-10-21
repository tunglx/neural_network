import random
import math
from active_function import ReLU_function
from layer import Layer
from cost_function import sum_squared_error

class NeuralNetwork:
    LEARNING_RATE = 0.1
    def __init__(self):
        self.layers = []

    def make_layer(self, num_neuron):
        if len(self.layers) == 0:
            num_weight = 0
        else:
            num_weight = len(self.layers[-1].neurons)

        layer = Layer(num_neuron, num_weight)
        self.layers.append(layer)

    def feed_forward(self, inputs):
        count = 0
        outputs = []
        for i in range(len(self.layers[count].neurons)):
            self.layers[count].neurons[i].output = inputs[i]
            outputs.append(inputs[i])

        count += 1
        while count < len(self.layers):
            outputs = self.layers[count].feed_forward(outputs)
            count += 1
            print outputs


    def total_error(self, training_outputs):
        total = 0
        outputs = self.layers[-1].get_outputs()
        for i in range(len(outputs)):
            total += sum_squared_error(outputs[i], training_outputs[i])
        return total

     
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        #1. Output neuron deltas
        for o in range(len(self.layers[-1].neurons)):
            self.layers[-1].neurons[o].delta = self.layers[-1].neurons[o].calc_delta(training_outputs[0])

        #2.Hidden neuron deltas
        num_layer = len(self.layers) - 2
        while num_layer > 0:
            layer = self.layers[num_layer]
            

            num_layer -= 1



        return 0
    def calculate_error(self, training_sets):
        return 0


nn = NeuralNetwork() 
nn.make_layer(5)
nn.make_layer(3)
nn.make_layer(3)
nn.make_layer(5)
nn.make_layer(2)

nn.train([1, 2, 3, 4, 5], [10, 20])
print nn.total_error([10, 20])

