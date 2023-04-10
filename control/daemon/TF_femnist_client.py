#!/usr/bin/env python3
# -*- coding: utf-8

import argparse
from _testmultiphase import error

import numpy as np
import os
import sys
import random
import tensorflow as tf
import flwr as fl

import json
from time import time

from cnn import ClientModel

ACCURACY_KEY = 'accuracy'

MODEL_PARAMS = (0.0003, 62)  # lr, num_classes
"""dict: Model specific parameter specification"""


STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'


def get_initial_weights(ckpt_file):
    try:
        with open(ckpt_file, "r") as control_file:
            lines = control_file.readlines()
            file = lines[-1]
        print("file to restore ", file)
        aux_data = np.load(file)
        print("removing ", file, " file")
        os.remove(file)
        aux_list: fl.common.NDArrays = []
        for file in aux_data.files:
            aux_list.append(aux_data[file])
        weights = aux_list
        return weights
    except Exception as e:
        print("ERROR:", e)
        return None


def get_args():
    # Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Flower client of LEAF with FEMINIST dataset')

    parser.add_argument('--batch-size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--seed',
                        help='seed for random client sampling and batch splitting',
                        type=int,
                        default=0)
    parser.add_argument('--metrics-name',
                        help='name for metrics file;',
                        type=str,
                        default='metrics',
                        required=False)
    parser.add_argument('--metrics-dir',
                        help='dir for metrics file;',
                        type=str,
                        default='metrics',
                        required=False)
    parser.add_argument('--use-val-set',
                        help='use validation set;',
                        action='store_true')
    parser.add_argument('-predst',
                        help='use validation set;',
                        type=str,
                        dest='folder_data',
                        default='bucket_folder')

    # Minibatch doesn't support num_epochs, so make them mutually exclusive
    epoch_capability_group = parser.add_mutually_exclusive_group()
    epoch_capability_group.add_argument('--num-epochs',
                                        help='number of epochs when clients train on data;',
                                        type=int,
                                        default=100)

    parser.add_argument('-lr',
                        help='learning rate for local optimizers;',
                        type=float,
                        default=-1,
                        required=False)

    parser.add_argument('-id_client',
                        help='Client identification',
                        required=True,
                        type=int,
                        default=-1)

    # Flower args
    flwr_args = parser.add_argument_group('Flower args')
    arg_groups.append(flwr_args)

    flwr_args.add_argument("-server_address", dest='server_address', type=str, required=True,
                           help=f"gRPC server address", default='localhost:8080')

    # Checkpoint args
    ckpt_args = parser.add_argument_group('Checkpoint args')
    arg_groups.append(ckpt_args)
    ckpt_args.add_argument("--save_ckpt", action="store_true", dest="save_ckpt", default=False,
                           help="Save checkpoint on client side")
    ckpt_args.add_argument("--restore_ckpt", action="store_true", dest="restore_ckpt", default=False,
                           help="Restore checkpoint from client side")
    ckpt_args.add_argument("-ckpt_file", dest="ckpt_file", type=str, default='checkpoints.txt',
                           help="Checkpoint file")

    return parser.parse_known_args()


def read_data(train_data_path, test_data_path):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    with open(train_data_path, 'r') as inf:
        data = json.load(inf)
    train_user = data['user']
    train_data = data['user_data']

    with open(test_data_path, 'r') as inf:
        data = json.load(inf)
    test_user = data['user']
    test_data = data['user_data']

    assert train_user == test_user

    return train_user, train_data, test_data


# Flower client
class CNNClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_data={'x': [], 'y': []}, eval_data={'x': [], 'y': []},
                 model: ClientModel = None, args=None):
        self._model = model
        self.id = client_id
        self.train_data = train_data
        self.eval_data = eval_data
        print("args.save_ckpt", args.save_ckpt)
        print("args.restore_ckpt", args.restore_ckpt)
        print("args.ckpt_file", args.ckpt_file)
        self.save_ckpt = args.save_ckpt
        self.restore_ckpt = args.restore_ckpt
        self.ckpt_file = args.ckpt_file
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        if self.restore_ckpt:
            initial_weights = get_initial_weights(self.ckpt_file)
            if initial_weights is not None:
                self._model.set_params(initial_weights)

    def get_parameters(self):
        return self._model.get_params()

    def fit(self, parameters, config):
        timeout = int(config["timeout"]) if "timeout" in config else None

        time_start = time()

        self._model.set_params(parameters)

        # Get hyperparameters for this round
        epochs = self.num_epochs
        batch_size = self.batch_size

        _, num_train_samples, parameters_prime = self.train(num_epochs=epochs)

        time_end = time()

        fit_duration = time_end - time_start
        if timeout is not None:
            if fit_duration > timeout:
                parameters_prime = []

        history = self.test(set_to_use='train')

        results = {
            "loss": float(history["loss"]),
            "accuracy": history[ACCURACY_KEY],
            "num_examples_ceil": num_train_samples*epochs,
            "num_examples": num_train_samples*epochs,
            "fit_duration": fit_duration
        }
        print("num_examples fit:", num_train_samples)
        print("fit results: ", results)
        return parameters_prime, num_train_samples, results

    def evaluate(self, parameters, config):
        # Update local model with global parameters
        self.model.set_params(parameters)

        if self.save_ckpt:
            global_epoch = int(config['epoch_global'])
            if global_epoch > 1:
                print(f"Removing old round-{global_epoch-1}-weights.npz saved file")
                try:
                    os.remove(f"round-{global_epoch-1}-weights.npz")
                except Exception as e:
                    print("ERROR:", e)
            print(f"Saving round {global_epoch} aggregated_ndarrays...")
            checkpoint_file = f"round-{global_epoch}-weights.npz"
            np.savez(checkpoint_file, *parameters)
            with open(self.ckpt_file, "a") as file:
                file.write(f"\n{checkpoint_file}")

        history = self.test()
        # # Evaluate global model parameters on the local test data and return results
        loss = float(history['loss'])
        accuracy = history[ACCURACY_KEY]
        print("num_examples test:", self.num_test_samples)
        print("evaluate loss: ", loss)
        print("evaluate accuracy: ", accuracy)
        return loss, self.num_test_samples, {"accuracy": accuracy}

    def train(self, num_epochs=100, batch_size=10):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """
        data = self.train_data
        comp, update = self.model.train(data, num_epochs, batch_size)

        num_train_samples = len(data['y'])
        return comp, num_train_samples, update

    def test(self, set_to_use='test'):
        """Tests self.model on self.test_data.

        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test' or set_to_use == 'val':
            data = self.eval_data
        return self.model.test(data)

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0
        if self.eval_data is not None:
            test_size = len(self.eval_data['y'])
        return train_size + test_size

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        self._model = model


if __name__ == "__main__":
    args, unparsed = get_args()

    # Set the random seed if provided (affects batching)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    tf.set_random_seed(123 + args.seed)

    # Suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)

    # Create 2 models
    model_params = MODEL_PARAMS
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Create client model, and share params with server model
    tf.reset_default_graph()
    client_model = ClientModel(args.seed, *model_params)

    eval_set = 'test' if not args.use_val_set else 'val'
    train_data_dir = os.path.join(args.folder_data, 'train')
    train_data_file = os.path.join(train_data_dir, 'train_data_user_' + str(args.id_client) + '.json')
    test_data_dir = os.path.join(args.folder_data, eval_set)
    test_data_file = os.path.join(test_data_dir, 'test_data_user_' + str(args.id_client) + '.json')

    user, train_data, test_data = read_data(train_data_file, test_data_file)

    client = CNNClient(client_id=user, train_data=train_data, eval_data=test_data,
                               model=client_model, args=args)
    fl.client.start_numpy_client(args.server_address, client=client)
