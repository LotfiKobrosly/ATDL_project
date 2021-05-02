import os
import json
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nets import NeuralNet, ConvNet
from utils import train_model

NETWORK_CHOICE = ["LeNet", "Convolutional_2", "Convolutional_4", "Convolutional_6"]

NETWORKS = {
    "LeNet": NeuralNet,
    "Convolutional_2": ConvNet,
    "Convolutional_4": ConvNet,
    "Convolutional_6": ConvNet
}

NETS_FILES = "../networks/networks.json"
OPTIMAL_PARAMS_PATH = "../networks/optimal_params.json"


def load_model(data_params, choice):
    with open(NETS_FILES, "r") as json_file:
        nets = json.load(json_file)
    params = nets[NETWORK_CHOICE[choice - 1]]
    for key in data_params.keys():
        params[key] = data_params[key]
    return NETWORKS[NETWORK_CHOICE[choice - 1]](**params)


def get_optima_hyperparams(choice):
    with open(OPTIMAL_PARAMS_PATH, "r") as json_file:
        all_params = json.load(json_file)
    return all_params[NETWORK_CHOICE[model_choice]]



if __name__ == "__main__":
    # Setting device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    device = torch.device(device)

    # Data choice
    out_message = """
Which dataset to test on?
1: MNIST
2: CIFAR10
    """

    data_choice = "p"
    while not (isinstance(data_choice, int)) or (data_choice not in [1, 2]):
        data_choice = str(input(out_message))
        try:
            data_choice = int(data_choice)
        except:
            data_choice = "p"

    if data_choice == 1:
        train_set = datasets.MNIST(root='../data',
                                   train=True,
                                   download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
        test_set = datasets.MNIST(root='../data',
                                  train=False,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))
        classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.CIFAR10(root='../data',
                                     train=True,
                                     download=True,
                                     transform=transform)
        test_set = datasets.CIFAR10(root='../data',
                                    train=False,
                                    download=True,
                                    transform=transform)
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train_data = DataLoader(train_set)
    _, (example, _) = next(enumerate(train_data))
    data_params = {
        "input_dim": list(example[0].size()),
        "output_dim": len(classes),
        "fc" : F.softmax}

    # Defining model and moving to device
    out_message = """
Which model do you choose?
1: LeNet
2: Convolutional 2
3: Convolutional 4
4: Convolutional 6
    """
    model_choice = "p"
    while not(isinstance(model_choice, int)) or (model_choice not in [1, 2, 3, 4]):
        model_choice = str(input(out_message))
        try:
            model_choice = int(model_choice)
        except:
            model_choice = "p"

    if model_choice == 1:
        data_params["input_dim"] = list(torch.reshape(example[0], (-1,)).size())[0]
    else:
        data_params["n_channels"] = list(example[0].size())[0]
    print(data_params)
    model = load_model(data_params, model_choice)
    model.set_device(device)

    # Get optimal parameters cited in the article
    optimal_params = get_optima_hyperparams(model_choice)
    pruning_rate = optimal_params.pop("pruning_rate")

    # Training and monitoring the training
    remaining = 1
    early_stopping_iter = list()
    train_acc = list()
    test_acc = list()
    while remaining > 0.05:
        _, validation_history = train_model(
            model,
            train_set,
            **optimal_params
        )
        train_acc.append(validation_history["accuracy"])
        early_stopping_iter.append(len(validation_history['accuracy']))
        test_acc.append(model.evaluate(test_set))

        model.prune(pruning_rate)
        remaining *= pruning_rate





