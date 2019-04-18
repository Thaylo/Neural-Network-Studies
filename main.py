import numpy as np


def sigmoid(x, derivative=False):
    y = 1. / (np.exp(-x) + 1.)
    if derivative:
        return np.multiply(y, (1. - y))
    return y


def feed_forward(x, syn):
    x_temp = x
    for s in syn:
        y_temp = sigmoid(x_temp)
        x_temp = np.dot(s.T, y_temp)
    return x_temp


def compute_gradient(x, y, syn):
    return 0


def loss(y, d):
    e = y-d
    return np.linalg.norm(e)


def main():

    np.random.seed(0)
    syn0 = np.random.rand(2, 4)
    syn1 = np.random.rand(4, 1)
    syn = [syn0, syn1]

    input_data = np.fromfile("data/input", sep=' ').reshape(4, 2)
    d = np.fromfile("data/output", sep=' ').reshape(4, 1)

    print(d)
    y = np.zeros(d.shape)

    for i in range(d.shape[0]):
        y[i] = feed_forward(input_data[i, :].reshape(2, 1), syn)

    print(y)
    print(loss(y, d))


if __name__ == "__main__":
    main()
