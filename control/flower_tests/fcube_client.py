import argparse
from collections import OrderedDict

from pathlib import Path

import numpy as np
import pandas as pd

import flwr as fl
import torch
import torch.nn as nn

import torch.utils.data as data

from torchvision.datasets import MNIST

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Generated(MNIST):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, args=None):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train
        self.dataidxs = dataidxs

        if self.train:
            # self.data = pd.read_csv(Path(args.path_dataset, "X_train.csv"), header=None, index_col=None).to_numpy()
            # self.targets = pd.read_csv(Path(args.path_dataset, "y_train.csv"), header=None, index_col=None).to_numpy()
            # self.targets = self.targets.flatten()
            self.data = np.load(Path(args.path_dataset, "X_train.npy"))
            self.targets = np.load(Path(args.path_dataset, "y_train.npy"))
            # print("y_train", self.targets)
            # print("y_train[0]", self.targets[1])
            # print("type of y_train", type(self.targets))
            # print("y_train.shape()", self.targets.shape())
        else:
            # self.data = pd.read_csv(Path(args.path_dataset, "y_test.csv"), header=None, index_col=None).to_numpy()
            # self.targets = pd.read_csv(Path(args.path_dataset, "y_test.csv"), header=None, index_col=None).to_numpy()
            self.data = np.load(Path(args.path_dataset, "X_test.npy"))
            self.targets = np.load(Path(args.path_dataset, "y_test.npy"))

        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.targets = self.targets[self.dataidxs]

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        return data, target

    def __len__(self):
        return len(self.data)


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


def main():
    """Create model, load data, define Flower client, start Flower client."""

    args = get_args()

    # Model (Perceptron)
    class PerceptronModel(nn.Module):
        def __init__(self, input_dim=3, output_dim=2):
            super(PerceptronModel, self).__init__()

            self.fc1 = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            x = self.fc1(x)
            return x

    nets = PerceptronModel()

    model_meta_data = []
    layer_type = []
    for (k, v) in nets.state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    # print("nets", nets)
    # print("model_meta_data", model_meta_data)
    # print("layer_type", layer_type)

    net = PerceptronModel().to(DEVICE)

    # Load data (FCUBE)
    trainloader, testloader, train_dataset, test_dataset = load_data(args)

    # Flower client
    class GeneratedClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)

            # Get hyperparameters for this round
            epochs = int(config["epochs"])
            lr = float(config["learning_rate"])
            rho = float(config["momentum"])
            reg = float(config["weight_decay"])

            loss, acc = train(net, trainloader, epochs=epochs, lr=lr, rho=rho, reg=reg)

            results = {
                "loss": loss,
                "accuracy": acc,
            }

            print("results", results)

            return self.get_parameters(), len(trainloader), results

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return loss, len(testloader), {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client(args.server_address, client=GeneratedClient())


def train(net, trainloader, epochs, lr, rho, reg):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=rho,
                                weight_decay=reg)
    net.train()
    if type(trainloader) == type([1]):
        pass
    else:
        trainloader = [trainloader]

    for e in range(epochs):
        epoch_loss_collector = []
        for tmp in trainloader:
            # print("tmp", tmp)
            # print("enumerate(tmp)", enumerate(tmp))
            # print("list(enumerate(tmp))", list(enumerate(tmp)))
            # for x, target in trainloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(DEVICE), target.to(DEVICE)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        print('Epoch: %d Loss: %f' % (e, epoch_loss))

        train_loss, train_acc = test(net, trainloader)

        return train_loss, train_acc


def test(net, testloader):
    """Validate the network on the entire test set."""
    net.eval()

    true_labels_list, pred_labels_list = np.array([]), np.array([])
    if type(testloader) == type([1]):
        pass
    else:
        testloader = [testloader]

    correct, total, loss = 0, 1, 0.0
    with torch.no_grad():
        for tmp in testloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(DEVICE), target.to(DEVICE, dtype=torch.int64)
                out = net(x)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if DEVICE == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
    accuracy = correct / total
    return loss, accuracy


def load_data(args):
    """Load FCUBE (training and test set)."""
    dl_obj = Generated

    train_ds = dl_obj(args.path_dataset, train=True, args=args)
    # TODO: divide test_ds into all parties for FCUBE
    test_ds = dl_obj(args.path_dataset, train=False, args=args)

    # print("train_ds", train_ds)
    # print("test_ds", test_ds)
    # print("args.batch_size", args.batch_size)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=args.batch_size,
                               shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=train_ds, batch_size=32,
                              shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds


if __name__ == "__main__":
    main()
