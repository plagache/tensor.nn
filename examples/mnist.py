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
        # print(f"{file_path} already exist")
        with open(file_path, "rb") as file:
            data = file.read()
    else:
        # print(f"Downloading {file_path}")
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(file_path, "wb") as file:
                data = response.content
                file.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8, offset=16) # but offset depends on the data labels 8, training 16


if __name__ == "__main__":
    # slicing the offset here
    x_train = fetch(google_url+data_sources["training_images"])[0x10:].reshape(-1, 28, 28)
    y_train = fetch(google_url+data_sources["training_labels"])[8:]
    x_test = fetch(google_url+data_sources["test_images"])[0x10:].reshape(-1, 28, 28)
    y_test = fetch(google_url+data_sources["test_labels"])[8:]
    x_train = x_train / 255
    x_train = x_train.astype(np.float32)

    # visualize data with matplolib and compare to label to verify our data
    # index_nbr = 55
    # plot_mnist(X_train[index_nbr], Y_train[index_nbr])

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

    # create class representing the model
    class neural_network():
        def __init__(self):
            self.layer_1 = Tensor(init_layer(784, hidden_size))
            self.layer_2 = Tensor(init_layer(hidden_size, number_of_output))

        def forward(self, x):
            output = x.dot(self.layer_1).relu().dot(self.layer_2)
            # output = x.dot(self.layer_1).relu().dot(self.layer_2).logsoftmax()
            return output


    batch_size = 100
    steps = 1000
    # learning_rate = 0.001
    learning_rate = Tensor(0.001)
    losses, accuracies = [], []
    for step in (t := tqdm(range(steps))):
        sample = np.random.randint(x_train.shape[0], size=batch_size)  # Randomly select batch_size samples
        model = neural_network()

        xs = Tensor(x_train[sample].reshape(-1, 28*28))

        # create the ys
        y_sampled = y_train[sample]
        ys = np.zeros((len(sample), number_of_label), np.float32)
        # encoding the correct label to its place [0,0,0,0,1,0,0,0,0,0]
        ys[range(ys.shape[0]), y_sampled] = 1.0
        # creating the tensor
        ys = Tensor(ys)


        output = model.forward(xs)
        # calculate the loss
        loss = output.mul(ys).mean()
        # loss = ((output - ys) ** 2).mean()
        loss.backward()

        # update parameters
        # model.layer_1.ndata = model.layer_1.ndata - np.multiply(learning_rate.ndata, model.layer_1.gradient.ndata)
        # model.layer_2.ndata = model.layer_2.ndata - np.multiply(learning_rate.ndata, model.layer_2.gradient.ndata)
        model.layer_1.ndata = model.layer_1.ndata - (learning_rate.ndata * model.layer_1.gradient.ndata)
        model.layer_2.ndata = model.layer_2.ndata - (learning_rate.ndata * model.layer_2.gradient.ndata)

        # Test accuracy
        # print(output.ndata)
        # print(np.info(output.ndata))
        # prediction = np.maximum(output.ndata, axis=1)
        # compare prediction and label to get accuracy
        # print(prediction)
        exit()

        t.set_description(f"loss: {loss}")
