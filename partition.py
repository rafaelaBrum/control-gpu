import os
import numpy as np

# MNIST Dataset
from keras.datasets import mnist


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def partition_data(dataset, n_parties):

    if dataset == 'MNIST':

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        n_train = y_train.shape[0]
        n_test = y_test.shape[0]

        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        idxs_test = np.random.permutation(n_test)
        batch_test_idxs = np.array_split(idxs_test, n_parties)
        net_test_dataidx_map = {i: batch_test_idxs[i] for i in range(n_parties)}
    else:
        print("No available dataset to partition.")
        return

    root_foldername = "data/{}/{}_clients".format(dataset, n_parties)
    mkdirs(root_foldername)
    dataset_file = f"{dataset}.npz"

    for i in range(n_parties):
        foldername = "{}/{}".format(root_foldername, i)
        mkdirs(foldername)
        print(f"created {foldername}")
        x_train_local = x_train[net_dataidx_map[i]]
        y_train_local = y_train[net_dataidx_map[i]]
        x_test_local = x_test[net_test_dataidx_map[i]]
        y_test_local = y_test[net_test_dataidx_map[i]]
        # print("x_train_local", x_train_local)
        # print("shape(x_train_local)", x_train_local.shape)
        # print("shape(x_train)", x_train.shape)
        np.savez(os.path.join(foldername, dataset_file), x_train=x_train_local, y_train=y_train_local,
                 x_test=x_test_local, y_test=y_test_local)


if __name__ == "__main__":
    dataset = 'MNIST'
    n_parties = 4
    partition_data(dataset, n_parties)
