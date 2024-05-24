import gzip
import hashlib
import os

import numpy as np
import requests
from fusion import Tensor

common_url = "http://yann.lecun.com/exdb/mnist/" # not accessible anymore
google_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
datasets_path = "static/datasets/"

data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "training_labels": "train-labels-idx1-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def fetch(url):
    file_path = os.path.join(datasets_path, hashlib.sha1(url.encode('utf-8')).hexdigest())
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(file_path, "wb") as file:
            data = response.content
            file.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8, offset=16) # but offset depends on the data labels 8, training 16


if __name__ == "__main__":
    X_train = fetch(google_url+data_sources["training_images"])[0x10:].reshape(-1, 28, 28)
    Y_train = fetch(google_url+data_sources["training_labels"])[8:]
    X_test = fetch(google_url+data_sources["test_images"])[0x10:].reshape(-1, 28, 28)
    Y_test = fetch(google_url+data_sources["test_labels"])[8:]
    print(X_train)
    print(np.info(X_train))
    print(Y_train)
    # visualize data with matplolib and compare to label to verify our data
