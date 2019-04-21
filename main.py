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
    input_data = np.random.rand(3, 1)
    nn.feed_forward(input_data)


if __name__ == "__main__":
    main()
