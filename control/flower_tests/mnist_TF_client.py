import argparse
# import os
# from collections import OrderedDict

import numpy as np

import flwr as fl
# import torch
# # import torch.nn as nn
# #
# import torch.utils.data as data

# from PIL import Image
# import torch.nn.functional as F
# import torchvision.transforms as transforms
from tensorflow import keras
from tensorflow.keras import layers

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description="Testando criar o servidor para FCUBE automaticamente")
    parser.add_argument(
        "--server_address", type=str, required=True,
        help=f"gRPC server address",
    )
    parser.add_argument(
        "--path_dataset", type=str, required=True,
        help=f"Path to dataset",
    )
    parser.add_argument(
        "--batch-size", type=int, required=True,
        help=f"Path to dataset",
    )
    args = parser.parse_args()
    return args


# Flower client
class MNISTClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        epochs = int(config["epochs"])
        # lr = float(config["learning_rate"])
        # rho = float(config["momentum"])
        # reg = float(config["weight_decay"])

        # loss, acc = train(self.model, trainloader, epochs=epochs, lr=lr, rho=rho, reg=reg)
        history = self.model.fit(
            self.x_train,
            self.y_train,
            32,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=5)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main():
    """Create model, load data, define Flower client, start Flower client."""

    args = get_args()

    num_classes = 10
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Start Flower client
    client = MNISTClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client(args.server_address, client=client)


def load_data(args):
    """Load MNIST (training and test set)."""
    #

    train_dl = [args.path_dataset]
    test_dl = []
    train_ds = []
    test_ds = []

    return train_dl, test_dl, train_ds, test_ds


if __name__ == "__main__":
    main()
