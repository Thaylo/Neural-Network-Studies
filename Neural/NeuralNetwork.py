import numpy as np


class NeuralLayer:
    input_tensor = []
    weights = []
    output_tensor = []

    def __init__(self, input_dimension, weights_dimension, output_dimension):
        self.input_tensor = np.random.rand(input_dimension[0], input_dimension[1])
        self.weights = np.random.rand(weights_dimension[0], weights_dimension[1])
        self.output_tensor = np.random.rand(output_dimension[0], output_dimension[1])

    def input_x(self, input_data):
        assert(input_data.shape == self.input_tensor.shape)
        self.input_tensor = input_data.copy()

    def update_state(self):
        y = np.dot(self.weights, self.input_tensor)
        assert(y.shape == self.output_tensor.shape)
        self.output_tensor = y
        return y


class NeuralNetwork:
    layers = []

    def __init__(self, nn_configuration_list):

        for nn_configuration in nn_configuration_list:
            input_dimension = nn_configuration[0]
            weights_dimension = nn_configuration[1]
            output_dimension = nn_configuration[2]
            self.layers.append(NeuralLayer(input_dimension, weights_dimension, output_dimension))

    def feed_forward(self, input_data):
        current_input = input_data

        for layer in self.layers:
            layer.input_x(current_input)
            current_input = layer.update_state()

        final_value = current_input

        return final_value


def sigmoid(x, derivative=False):
    y = 1. / (np.exp(-x) + 1.)

    if derivative:
        return np.multiply(y, (1. - y))

    return y
