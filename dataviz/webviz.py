import argparse
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as pyplot
# from bokeh.resources import CDN
from bokeh.plotting import figure, show
import holoviews as hv
import hvplot.pandas
from bokeh.sampledata.penguins import data as df
hv.extension('bokeh')

df.hvplot.scatter(x='bill_length_mm', y='bill_depth_mm', by='species')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
    parser.add_argument("dataset", help="the dataset csv file")
    args = parser.parse_args()

    # print(args.dataset)
    dataset = pl.read_csv(args.dataset)

    # creating a new columns ys with 1 representing Malign and 0 representing Benign Cells
    dataset = dataset.with_columns(pl.col("M").map_elements(lambda x: 1 if x == "M" else 0, float).alias("ys"))
    # dataset = dataset.select(pl.col(pl.NUMERIC_DTYPES))
    dataset = dataset.select(cs.numeric())

    dataset.plot.scatter()
    print(type(dataset.plot.scatter()))
    print(type(dataset.plot))
    # hvplot.extension('matplotlib')
    # hvplot.extension('plotly')
    # fig = dataset.plot.scatter()
    # fig = px.scatter(dataset)
    # features = dataset.columns
    # features_pairs = [(feature, list(filter(lambda x: x != feature, features))) for feature in features]
    # for given_feature, other_features in features_pairs:
    #     for other_feature in other_features:
    #         title = f"{given_feature} - {other_feature}"
    #         # pyplot.title(title)
    #         # set xylabels
    #         px.scatter(
    #             dataset[given_feature],
    #             dataset[other_feature],
    #         )
    #         px.legend(loc="best")
    #         px.show()
            # pyplot.legend(loc="best")
            # pyplot.show()
    exit()
    for column in dataset.columns:
        fig = pyplot.scatter(dataset)
        fig.show()

    print(dataset)
    print(dataset.describe())
    print(dataset)
