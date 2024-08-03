import argparse
import polars as pl
import polars.selectors as cs
from fusion import Tensor
import numpy as np


hidden_layer = 10
output_layer = 2


def init_layer(m, h):
    layer = np.random.uniform(-1.0, 1.0, size=(m, h))
    return layer.astype(np.float32)


class Model:
    def __init__(self):
        # Params, so no grad
        self.input_layer = Tensor(init_layer(number_of_features, hidden_layer))
        self.hidden_layer_1 = Tensor(init_layer(hidden_layer, hidden_layer))
        # self.hidden_layer_2 = Tensor(init_layer(hidden_layer))
        self.output_layer = Tensor(init_layer(hidden_layer, output_layer))

    def forward(self, inputs):
        output = inputs.dot(self.input_layer).relu().dot(self.hidden_layer_1).relu().dot(self.output_layer).relu()
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
    parser.add_argument("dataset", help="the dataset csv file")
    args = parser.parse_args()

    # print(args.dataset)
    dataset = pl.read_csv(args.dataset)
    # numerical_dataset = dataset.select(pl.col(pl.NUMERIC_DTYPES))
    numerical_dataset = dataset.select(cs.numeric())

    number_of_inputs = dataset["M"].len()
    number_of_features = numerical_dataset.width
    print(f"number_of_input: {number_of_inputs}")
    print(f"number_of_features: {number_of_features}")
    # print(type(number_of_inputs))

    # creating a new columns ys with 1 representing Malign and 0 representing Benign Cells
    dataset = dataset.with_columns(pl.col("M").map_elements(lambda x: 1 if x == "M" else 0, float).alias("ys"))
    labels = dataset["M"]

    xs = numerical_dataset
    # print(xs)
    xs = Tensor(xs.to_numpy())
    # print(f"tensor(xs): {xs}")

    ys = dataset["ys"]
    # print(labels)
    # print(f"ys: {ys}")


    model = Model()
    # output = model.forward(xs.T)
    output = model.forward(xs)
    backward = output.backward()
    # output = model.forward(xs.T).logsoftmax()
    print(output)
    # print(backward)
    print(model.input_layer.gradient)
    # print(dataset)
    # print(dataset.describe())
    # print(numerical_dataset)
