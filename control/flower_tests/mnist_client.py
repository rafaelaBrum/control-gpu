import argparse
import os
from collections import OrderedDict

import numpy as np

import flwr as fl
import torch
import torch.nn as nn

import torch.utils.data as data

from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MNIST_truncated(data.Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            self.data, self.target = torch.load(os.path.join(root, 'training.pt'))
        else:
            self.data, self.targets = torch.load(os.path.join(root, 'test.pt'))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]


        # print("img", img)
        # print("type(img)", type(img))


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        # print("mnist img:", img)
        # print("mnist target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

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

    # Model (SimpleCNN)
    class SimpleCNNMNIST(nn.Module):
        def __init__(self, input_dim, hidden_dims, output_dim=10):
            super(SimpleCNNMNIST, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)

            # for now, we hard coded this network
            # i.e. we fix the number of hidden layers i.e. 2 layers
            self.fc1 = nn.Linear(input_dim, hidden_dims[0])
            self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
            self.fc3 = nn.Linear(hidden_dims[1], output_dim)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 4 * 4)

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).to(DEVICE)

    # Load data (MNIST)
    trainloader, testloader, train_dataset, test_dataset = load_data(args)

    # Flower client
    class MNISTClient(fl.client.NumPyClient):
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

            # print("results", results)

            return self.get_parameters(), len(trainloader), results

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return loss, len(testloader), {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client(args.server_address, client=MNISTClient())


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
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()

    true_labels_list, pred_labels_list = np.array([]), np.array([])
    if type(testloader) == type([1]):
        pass
    else:
        testloader = [testloader]

    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for tmp in testloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(DEVICE), target.to(DEVICE, dtype=torch.int64)
                out = net(x)
                loss += criterion(out, target).item()
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

    accuracy = correct / float(total)
    return loss, accuracy


def load_data(args):
    """Load MNIST (training and test set)."""
    dl_obj = MNIST_truncated

    transform_train = transforms.ToTensor()

    transform_test = transforms.ToTensor()

    train_ds = dl_obj(args.path_dataset, train=True, transform=transform_train)
    test_ds = dl_obj(args.path_dataset, train=False, transform=transform_test)

    # print("train_ds", train_ds)
    # print("test_ds", test_ds)
    # print("args.batch_size", args.batch_size)
    # print("type(data)", type(train_ds))
    # print("type(target)", type(test_ds))

    train_dl = data.DataLoader(dataset=train_ds, batch_size=args.batch_size,
                               shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=train_ds, batch_size=32,
                              shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds


if __name__ == "__main__":
    main()
