import argparse
import polars as pl
import polars.selectors as cs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
    parser.add_argument("dataset", help="the dataset csv file")
    args = parser.parse_args()

    # print(args.dataset)
    dataset = pl.read_csv(args.dataset)

    # creating a new columns ys with 1 representing Malign and 0 representing Benign Cells
    dataset = dataset.with_columns(pl.col("M").map_elements(lambda x: 1 if x == "M" else 0, float).alias("ys"))
    # numerical_dataset = dataset.select(pl.col(pl.NUMERIC_DTYPES))
    numerical_dataset = dataset.select(cs.numeric())

    print(dataset)
    print(dataset.describe())
    print(numerical_dataset)
