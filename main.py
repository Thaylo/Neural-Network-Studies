import numpy as np

import Neural.NeuralNetwork as nN


def main():

    np.random.seed(0)

    nn_configuration = []

    layer_config_a = [(3, 1), (4, 3), (4, 1)]
    layer_config_b = [(4, 1), (1, 4), (1, 1)]

    nn_configuration.append(layer_config_a)
    nn_configuration.append(layer_config_b)

    nn = nN.NeuralNetwork(nn_configuration)

    input_filename = "data/input"
    output_filename = "data/output"
    input_data = np.fromfile(input_filename, sep=' ').reshape(3, 4)
    output_data = np.fromfile(output_filename, sep=' ').reshape(1, 4)

    for i in range(4):
        x = input_data[:, i].reshape(3, 1)
        d = output_data[:, i].reshape(1, 1)
        nn.compute_gradient(x, d)


if __name__ == "__main__":
    main()
