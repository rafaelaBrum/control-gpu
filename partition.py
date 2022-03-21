import os
from typing import List

import numpy as np
from shutil import copyfile

# MNIST Dataset
from keras.datasets import mnist

# Local imports
from control.cnn_app.PImage import PImage


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def _load_metadata_from_dir(d):
    """
    Create SegImages from a directory
    """
    class_set = set()
    labels = open(os.path.join(d, 'label.txt'), 'r')

    t_x, t_y = ([], [])
    for f in labels:
        tmp = f.strip().split()
        f_name, f_label = tmp[0], tmp[1]
        origin = ''
        coord = None
        label = int(f_label)
        if label < 1:
            label = 0
        if len(tmp) > 2:
            origin = tmp[2]
        if len(tmp) > 4:
            coord = (tmp[3], tmp[4])
        t_path = os.path.join(d, f_name)
        if os.path.isfile(t_path):
            seg = PImage(t_path, keep_img=False, origin=origin, coord=coord, verbose=1)
            t_x.append(seg)
            t_y.append(label)
            class_set.add(label)
        else:
            print("Label file contains reference to {0}, but no such file exists.".format(t_path))

    # Non-lymphocyte patches are labeld 0 or -1 (no lymphocyte or below lymphocyte threshold)
    # -1 and 0 labels are treated as the same as for now this is a binary classification problem
    print("On directory {2}:\n - Number of classes: {0};\n - Classes: {1}"
          "\nNumber of samples: {3}".format(len(class_set), class_set, d, len(t_x)))

    return t_x, t_y


def copy_images(x_train: List[PImage], new_path):
    for im in x_train:
        old_path = im.get_path()
        image_name = im.get_img_name()
        # print("str(im)", str(im))
        # print("old_path", old_path)
        # print("image_name", image_name)
        image_path = os.path.join(new_path, image_name + ".png")
        copyfile(old_path, image_path)


def generate_label_files(tdir):
    """
    CellRep datasources use label text files to store ground truth for each patch.
    File: label.txt
    Format: file_name label source_svs x y
    """

    for d in os.listdir(tdir):
        c_dir = os.path.join(tdir, d)
        if os.path.isdir(c_dir):
            patches = os.listdir(c_dir)
            patches = list(filter(lambda p: p.endswith('.png'), patches))
            with open(os.path.join(c_dir, 'label.txt'), 'w') as fd:
                for im in patches:
                    fields = im.split('.')[0].split('_')
                    label = fields[1]
                    fields = fields[0].split('-')
                    fields = [im, label, '-'.join(fields[:6]), fields[6], fields[7]]
                    fd.write("{}\n".format(" ".join(fields)))

    print("Done generating label files")


def partition_MNIST_data(dataset, n_parties):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    # random partition
    idxs = np.random.permutation(n_train)
    batch_idxs = np.array_split(idxs, n_parties)
    net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
    idxs_test = np.random.permutation(n_test)
    batch_test_idxs = np.array_split(idxs_test, n_parties)
    net_test_dataidx_map = {i: batch_test_idxs[i] for i in range(n_parties)}
    # print("type(net_dataidxs_map)", type(net_dataidx_map))
    # print("type(net_test_dataidxs_map)", type(net_test_dataidx_map))

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


def partition_CellRep_data(dataset, n_parties):
    trainset_folder = os.path.join(os.path.expanduser('~'), 'IMGs-EN-194', 'trainset', 'samples')
    x_train_list, y_train_list = _load_metadata_from_dir(trainset_folder)
    testset_folder = os.path.join(os.path.expanduser('~'), 'IMGs-EN-194', 'testset', 'samples')
    x_test_list, y_test_list = _load_metadata_from_dir(testset_folder)

    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)

    n_train = len(y_train)
    n_test = len(y_test)

    # Dirichlet label partition
    min_size = 0
    min_size_test = 0
    min_require_size = 5
    min_test_require_size = 5
    K = 2
    beta = 0.9

    N = n_train
    N_test = n_test
    np.random.seed(2020)
    net_dataidx_map = {}
    net_test_dataidx_map = {}

    while (min_size < min_require_size) and (min_size_test < min_test_require_size):
        idx_batch = [[] for _ in range(n_parties)]
        idx_test_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            idx_test_k = np.where(y_test == k)[0]
            np.random.shuffle(idx_k)
            np.random.shuffle(idx_test_k)
            # proportions = np.repeat(1/n_parties, n_parties)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            # proportions_test = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions_test = proportions
            print("proportions1: ", proportions)
            print("sum pro1:", np.sum(proportions))
            print("proportions1_test: ", proportions_test)
            print("sum pro1_test:", np.sum(proportions_test))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions_test = np.array([p * (len(idx_j) < N_test / n_parties) for p, idx_j in zip(proportions_test, idx_test_batch)])
            # print("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            proportions_test = proportions_test / proportions_test.sum()
            # print("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            proportions_test = (np.cumsum(proportions_test) * len(idx_test_k)).astype(int)[:-1]
            # print("proportions4: ", proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            idx_test_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_test_batch, np.split(idx_test_k, proportions_test))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            min_size_test = min([len(idx_j) for idx_j in idx_test_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            np.random.shuffle(idx_test_batch[j])
            net_test_dataidx_map[j] = idx_test_batch[j]



    root_foldername = "data/{}/{}_clients".format(dataset, n_parties)
    mkdirs(root_foldername)

    for party in range(n_parties):
        foldername = "{}/{}".format(root_foldername, party)
        mkdirs(foldername)
        print(f"created {foldername}")
        # saving train dataset
        x_train_local = []
        for i in range(len(x_train_list)):
            if i in net_dataidx_map[party]:
                x_train_local.append(x_train_list[i])
        foldername = f"{foldername}/trainset/samples/"
        mkdirs(foldername)
        print(f"created {foldername}")
        copy_images(x_train_local, foldername)
        foldername = "{}/{}/trainset".format(root_foldername, party)
        generate_label_files(foldername)
        # saving test dataset
        x_test_local = []
        for i in range(len(x_test_list)):
            if i in net_test_dataidx_map[party]:
                x_test_local.append(x_train_list[i])
        foldername = f"{root_foldername}/{party}/testset/samples/"
        mkdirs(foldername)
        print(f"created {foldername}")
        copy_images(x_test_local, foldername)
        foldername = "{}/{}/testset".format(root_foldername, party)
        generate_label_files(foldername)


def partition_patches_data_homogeneously(dataset, n_parties, folder):
    dataset_folder = os.path.join(os.path.expanduser('~'), dataset, folder, 'samples')
    x_list, y_list = _load_metadata_from_dir(dataset_folder)

    y_array = np.array(y_list)

    n_dataset = len(y_array)

    # random partition
    idxs = np.random.permutation(n_dataset)
    batch_idxs = np.array_split(idxs, n_parties)
    net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    root_foldername = "{}/{}/{}_clients".format(dataset, folder, n_parties)
    mkdirs(root_foldername)

    for party in range(n_parties):
        foldername = "{}/{}".format(root_foldername, party)
        mkdirs(foldername)
        print(f"created {foldername}")
        # saving train dataset
        x_local = []
        for i in range(len(x_list)):
            if i in net_dataidx_map[party]:
                x_local.append(x_list[i])
        foldername = f"{foldername}/samples/"
        mkdirs(foldername)
        print(f"created {foldername}")
        copy_images(x_local, foldername)
        foldername = "{}/{}".format(root_foldername, party)
        generate_label_files(foldername)


def partition_data_homogeneously(dataset, n_parties, folder):
    trainset_folder = os.path.join(os.path.expanduser('~'), dataset, folder, 'trainset', 'samples')
    x_train_list, y_train_list = _load_metadata_from_dir(trainset_folder)
    testset_folder = os.path.join(os.path.expanduser('~'), dataset, folder, 'testset', 'samples')
    x_test_list, y_test_list = _load_metadata_from_dir(testset_folder)

    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)

    n_train = len(y_train)
    n_test = len(y_test)

    # random partition
    idxs = np.random.permutation(n_train)
    batch_idxs = np.array_split(idxs, n_parties)
    net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
    idxs_test = np.random.permutation(n_test)
    batch_idxs_test = np.array_split(idxs_test, n_parties)
    net_test_dataidx_map = {i: batch_idxs_test[i] for i in range(n_parties)}

    root_foldername = "data/{}/{}_clients".format(dataset, n_parties)
    mkdirs(root_foldername)

    for party in range(n_parties):
        foldername = "{}/{}".format(root_foldername, party)
        mkdirs(foldername)
        print(f"created {foldername}")
        # saving train dataset
        x_train_local = []
        for i in range(len(x_train_list)):
            if i in net_dataidx_map[party]:
                x_train_local.append(x_train_list[i])
        foldername = f"{foldername}/trainset/samples/"
        mkdirs(foldername)
        print(f"created {foldername}")
        copy_images(x_train_local, foldername)
        foldername = "{}/{}/trainset".format(root_foldername, party)
        generate_label_files(foldername)
        # saving test dataset
        x_test_local = []
        for i in range(len(x_test_list)):
            if i in net_test_dataidx_map[party]:
                x_test_local.append(x_train_list[i])
        foldername = f"{root_foldername}/{party}/testset/samples/"
        mkdirs(foldername)
        print(f"created {foldername}")
        copy_images(x_test_local, foldername)
        foldername = "{}/{}/testset".format(root_foldername, party)
        generate_label_files(foldername)


def partition_data(dataset, n_parties, folder):

    if dataset == 'MNIST':
        partition_MNIST_data(dataset, n_parties)
    elif dataset == 'CellRep':
        partition_CellRep_data(dataset, n_parties)
    elif dataset == 'patches':
        partition_patches_data_homogeneously(dataset, n_parties, folder)
    else:
        print("No available dataset to partition.")
        return


if __name__ == "__main__":
    # dataset = 'patches'
    # for folder in ('blca', 'brca', 'cesc', 'coad', 'paad', 'prad', 'read', 'skcm', 'stad', 'ucec'):
    #     for n_parties in (2, 3, 4):
    #         print(f"Partitioning folder {folder} into {n_parties} clients")
    #         partition_data(dataset, n_parties, folder)
    n_parties = 2
    dataset = 'client_0'
    folder = 'CellRep/4_clients/0'

    partition_data_homogeneously(dataset, n_parties, folder)