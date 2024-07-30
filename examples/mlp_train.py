import argparse
import polars as pl
import polars.selectors as cs
from fusion import Tensor
import numpy as np


hidden_size = 31
number_of_output = 1
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
class Model():
    def __init__(self):
        self.layer_1 = Tensor(init_layer(number_of_input, hidden_size))
        self.layer_2 = Tensor(init_layer(hidden_size, number_of_output))

    def forward(self, xs):
        output = xs.dot(self.layer_1).relu().dot(self.layer_2).relu()
        # output = x.dot(self.layer_1).relu().dot(self.layer_2).logsoftmax()
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
    parser.add_argument("dataset", help="the dataset csv file")
    args = parser.parse_args()

    # print(args.dataset)
    dataset = pl.read_csv(args.dataset)
    xs = dataset.drop("M")
    # print(xs)
    xs = Tensor(xs.to_numpy())
    print(f"tensor(xs): {xs}")
    number_of_input = dataset["M"].len()
    # print(f"number_of_input: {number_of_input}")
    # print(type(number_of_input))

    # creating a new columns ys with 1 representing Malign and 0 representing Benign Cells
    dataset = dataset.with_columns(pl.col("M").map_elements(lambda x: 1 if x == "M" else 0, float).alias("ys"))
    labels = dataset["M"]
    ys = dataset["ys"]
    # print(labels)
    print(f"ys: {ys}")

    # numerical_dataset = dataset.select(pl.col(pl.NUMERIC_DTYPES))
    numerical_dataset = dataset.select(cs.numeric())

    model = Model()
    output = model.forward(xs.T)
    grad = output.backward()
    # output = model.forward(xs.T).logsoftmax()
    print(output)
    # print(dataset)
    # print(dataset.describe())
    # print(numerical_dataset)
