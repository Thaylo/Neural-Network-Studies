import numpy as np
import matplotlib.pyplot as plt

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

    def optimize(self, input_data, output_data, generate_graph=False):
        error_historic = []
        iterations = []

        for iter in range(25000):
            for layer in self.layers:

                grad_w = np.zeros(layer.grad_w.shape)
                grad_bias = np.zeros(layer.grad_bias.shape)
                n_inputs = input_data.shape[1]

                for i in range(n_inputs):
                    x = input_data[:, i].reshape(2, 1)
                    d = output_data[:, i].reshape(1, 1)
                    self.compute_gradient(x, d)
                    grad_w += layer.grad_w/n_inputs
                    grad_bias += layer.grad_bias/n_inputs

                alpha = 0.15
                layer.weights -= alpha * grad_w
                layer.bias -= alpha * grad_bias

            if generate_graph:

                if iter % 1000 == 0:
                    iterations.append(iter)
                    e2 = np.zeros((1, 1))
                    for i in range(n_inputs):
                        x = input_data[:, i].reshape(2, 1)
                        d = output_data[:, i].reshape(1, 1)
                        err = self.feed_forward(x) - d
                        e2 += np.dot(err, err.T)/n_inputs
                    error_historic.append(e2[0, 0])

        if generate_graph:
            plt.plot(iterations, error_historic)
            plt.legend(['Quadratic error'])
            plt.show()
            plt.savefig("result.png")


def sigmoid(x, derivative=False):
    y = 1. / (np.exp(-x) + 1.)

    if derivative:
        return np.multiply(y, (1. - y))

    return y
