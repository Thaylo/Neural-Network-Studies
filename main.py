import numpy as np
import Neural.NeuralNetwork as nN


def main():

    np.random.seed(0)

    nn_configuration = []

    layer_config_a = [(2, 1), (4, 2), (4, 1)]
    layer_config_b = [(4, 1), (1, 4), (1, 1)]

    nn_configuration.append(layer_config_a)
    nn_configuration.append(layer_config_b)

    nn = nN.NeuralNetwork(nn_configuration)

    input_filename = "data/input"
    output_filename = "data/output"
    input_data = np.fromfile(input_filename, dtype=np.dtype('f8'), sep=' ').reshape(2, 4)
    output_data = np.fromfile(output_filename, dtype=np.dtype('f8'), sep=' ').reshape(1, 4)

    nn.optimize(input_data, output_data, True)

    for i in range(4):
        d = output_data[:, i].reshape(1, 1)
        print(d)

    for i in range(4):
        x = input_data[:, i].reshape(2, 1)
        y = nn.feed_forward(x)
        print(y)


if __name__ == "__main__":
    main()
