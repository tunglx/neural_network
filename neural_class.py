import random
import math

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    # Active function
    def reLU(self, total_net_input):
        if total_net_input <= 0:
            return 0
        else:
            return total_net_input

    # Calculate the value of neuron with inputs value
    def calc_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # Neuron value after taking activation function
    def calc_output(self, inputs):
        self.inputs = inputs
        self.output = self.reLU(self.calc_total_net_input())
        return self.output

class Layer:
    def __init__(self, num_neuron, bias):
        self.bias = bias if bias else random.random()
        self.neuron = []
        for i in range(num_neuron):
            self.neuron.append(Neuron(self.bias))

    def print_out_network(self):
        print 'Total neuron: ', len(self.neuron)

        for n in range(len(self.neuron)):
            print 'Neuron number: ', n
            for w in range(len(self.neuron[n].weights)):
                print 'weights:', self.neuron[n].weights[w]
            print 'Bias:', self.bias

    def feed_forward(self, inputs):
        outputs = []
        for neural in self.neuron:
            output.append(neural.calc_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neural in self.neuron:
            outputs.append(neural.calc_output(inputs))
        return outputs


class NeuralNetwork:
    LEARNING_RATE = 0.1
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs
        self.hidden_layer = Layer(num_hidden, hidden_layer_bias)
        self.output_layer = Layer(num_hidden, output_layer_bias)

        self.init_weight_from_inputs_to_hidden(hidden_layer_weights)
        self.init_weight_from_hidden_to_output(output_layer_weights)

    def init_weight_from_inputs_to_hidden(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neuron)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neuron[h].weights.append(random.random())
                else:
                    self.hidden_layer.neuron[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weight_from_hidden_to_output(self, output_layer_weights):
        weight_num = 0
        for h in range(len(self.output_layer.neuron)):
            for i in range(len(self.hidden_layer.neuron)):
                if not output_layer_weights:
                    self.output_layer.neuron[h].weights.append(random.random())
                else:
                    self.output_layer.neuron[h].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def print_out_network(self):
        print '------'
        print 'Inputs: ', self.num_inputs
        print'------'
        print '-Hidden Layer-'
        self.hidden_layer.print_out_network()
        print '------'
        print '-Output Layer-'
        self.output_layer.print_out_network()
        print '------'

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    def train(self, training_inputs, training_outputs):
        return 0
    def calculate_error(self, training_sets):
        return 0


nn = NeuralNetwork(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
nn.print_out_network()