import gzip
import hashlib
import os

import matplotlib.pyplot as pyplot
import numpy as np
import requests
from fusion import Tensor
from tqdm import tqdm

common_url = "http://yann.lecun.com/exdb/mnist/"  # not accessible anymore
google_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
datasets_path = "static/datasets/"

data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "training_labels": "train-labels-idx1-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

def plot_mnist(image, title):
    pyplot.imshow(image, cmap="gray")
    pyplot.title(title)
    pyplot.show()

def fetch(url):
    file_path = os.path.join(datasets_path, hashlib.sha1(url.encode('utf-8')).hexdigest())
    isExisting = os.path.exists(file_path)
    if isExisting is True:
        print(f"{file_path} already exist")
        with open(file_path, "rb") as file:
            data = file.read()
    else:
        print(f"Downloading {file_path}")
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(file_path, "wb") as file:
                data = response.content
                file.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8, offset=16) # but offset depends on the data labels 8, training 16


if __name__ == "__main__":
    x_train = fetch(google_url+data_sources["training_images"])[0x10:].reshape(-1, 28, 28)
    y_train = fetch(google_url+data_sources["training_labels"])[8:]
    x_test = fetch(google_url+data_sources["test_images"])[0x10:].reshape(-1, 28, 28)
    y_test = fetch(google_url+data_sources["test_labels"])[8:]
    index_nbr = 55
    # plot_mnist(X_train[index_nbr], Y_train[index_nbr])
    # print(X_train)
    # print(np.info(X_train))
    # convert 255 values into floating point
    x_train = x_train / 255
    # print(X_train[index_nbr])
    x_train = x_train.astype(np.float32)
    # print(X_train[index_nbr])
    # print(X_train)
    # print(np.info(X_train))
    # print(Tensor(X_train))
    # print(np.info(X_train))
    # print(Y_train)
    # print(np.info(Y_train))
    # visualize data with matplolib and compare to label to verify our data

    hidden_size = 128
    # the 2 next should be the same U_u
    number_of_output = 10
    number_of_label = 10
    # h: height of the layer
    # m: probably number of input for each neuron in the layer
    # this function initialize the weight/parameters of our layers
    def init_layer(m, h):
        initial_parameters = np.random.uniform(-1.0, 1.0, size=(m, h))

        # Scaling with Xavier Glorot initaliazation
        # layer = initial_parameters / np.sqrt(m * h)
        layer = initial_parameters

        return layer.astype(np.float32)

    # create class
    class neural_network():
        def __init__(self):
            self.layer_1 = Tensor(init_layer(784, hidden_size))
            self.layer_2 = Tensor(init_layer(hidden_size, number_of_output))

        def forward(self, x):
            output = x.dot(self.layer_1).relu().dot(self.layer_2)
            return output


    batch_size = 100
    steps = 1000
    learning_rate = 0.001
    losses, accuracies = [], []
    for step in (t := tqdm(range(steps))):
        sample = np.random.randint(x_train.shape[0], size=batch_size)  # Randomly select batch_size samples
        model = neural_network()

        output = model.forward(Tensor(x_train[sample].reshape(-1, 28*28)))
        # print("\noutput :", output)
        # print("\noutput.ndata :", output.ndata)
        # exit()

        # create the y
        y_sampled = y_train[sample]
        ys = np.zeros((len(sample), number_of_label), np.float32)
        ys[range(ys.shape[0]), y_sampled] = 1.0
        ys = Tensor(ys)
        # print("\nys.ndata :", ys.ndata)

        # loss = ((output - ys) ** 2).mean()
        # loss.backward()

        # calculate the loss
        loss = output.mul(ys).mean()
        # print(loss)
        # print(loss._context.parents)
        loss.backward()
        # print(output.gradient)
        # print(ys.gradient)

        # update the layer
        # print(model.layer_1)
        print(model.layer_1.gradient)
        # print(model.layer_2)
        print(model.layer_2.gradient)
        exit()
