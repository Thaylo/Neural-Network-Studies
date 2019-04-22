import numpy as np


class NeuralLayer:
    input_tensor = []
    weights = []
    bias = []
    output_tensor = []
    grad_w = []
    grad_bias = []

    def __init__(self, input_dimension, weights_dimension, output_dimension):
        self.input_tensor = np.random.rand(input_dimension[0], input_dimension[1])
        self.weights = np.random.rand(weights_dimension[0], weights_dimension[1])
        self.grad_w = np.random.rand(weights_dimension[0], weights_dimension[1])
        self.bias = np.random.rand(output_dimension[0], output_dimension[1])
        self.grad_bias = np.random.rand(output_dimension[0], output_dimension[1])
        self.output_tensor = np.random.rand(output_dimension[0], output_dimension[1])

    def input_x(self, input_data):
        assert(input_data.shape == self.input_tensor.shape)
        self.input_tensor = input_data.copy()

    def update_state(self):
        y = sigmoid(np.dot(self.weights, self.input_tensor) + self.bias)
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

    def compute_gradient(self, x, d):
        y = self.feed_forward(x)
        del_y = y - d

        for layer in reversed(self.layers):
            w = layer.weights
            y = layer.output_tensor
            y_prev = layer.input_tensor
            del_y_del_x = np.multiply(y, (1. - y))
            del_x = np.multiply(del_y, del_y_del_x)
            layer.grad_bias = del_x
            layer.grad_w = np.dot(del_x, y_prev.T)
            del_y_prev = np.dot(w.T, del_x)
            del_y = del_y_prev

    def optimize(self, input_data, output_data):
        for iter in range(5000):
            for i in range(4):
                x = input_data[:, i].reshape(2, 1)
                d = output_data[:, i].reshape(1, 1)
                self.compute_gradient(x, d)

                for layer in self.layers:
                    alpha = 0.3
                    layer.weights -= alpha * layer.grad_w
                    layer.bias -= alpha * layer.grad_bias


def sigmoid(x, derivative=False):
    y = 1. / (np.exp(-x) + 1.)

    if derivative:
        return np.multiply(y, (1. - y))

    return y
