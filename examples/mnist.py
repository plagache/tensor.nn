from fusion import Tensor

import gzip, requests

common_url = "http://yann.lecun.com/exdb/mnist/" # not accessible anymore
google_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
file_path = "static/Datasets/"

data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "training_labels": "train-labels-idx1-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

def download_data(url, filename):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(file_path+filename, "wb") as file:
            file.write(response.content)
    return response

def fetch_mnist():
    for name in data_sources:
        download_data(google_url+data_sources[name], name)
    return

if __name__ == "__main__":
    fetch_mnist()
