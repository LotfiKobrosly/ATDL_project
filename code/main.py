import os
import json
import ssl

import tkinter
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from nets import NeuralNet, ConvNet
from utils import train_model

# Settings
ssl._create_default_https_context = ssl._create_unverified_context

# Global variables
NETWORK_CHOICE = ["LeNet", "Convolutional_2", "Convolutional_4", "Convolutional_6"]

NETWORKS = {
    "LeNet": NeuralNet,
    "Convolutional_2": ConvNet,
    "Convolutional_4": ConvNet,
    "Convolutional_6": ConvNet,
}

DATA_CHOICES = ["MNIST", "CIFAR10"]

NETS_FILES = "../networks/networks.json"
OPTIMAL_PARAMS_PATH = "../networks/optimal_params.json"
FIGURES_PATH = "../figures/"


def load_model(data_params, choice):
    with open(NETS_FILES, "r") as json_file:
        nets = json.load(json_file)
    params = nets[NETWORK_CHOICE[choice - 1]]
    for key in data_params.keys():
        params[key] = data_params[key]
    return NETWORKS[NETWORK_CHOICE[choice - 1]](**params)


def get_optimal_hyperparams(choice):
    with open(OPTIMAL_PARAMS_PATH, "r") as json_file:
        all_params = json.load(json_file)
    return all_params[NETWORK_CHOICE[model_choice]]


if __name__ == "__main__":
    # Setting device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Data choice
    out_message = """
Which dataset to test on?
1: MNIST
2: CIFAR10
Answer: """

    data_choice = 2
    while not (isinstance(data_choice, int)) or (data_choice not in [1, 2]):
        data_choice = str(input(out_message))
        try:
            data_choice = int(data_choice)
        except:
            data_choice = "p"

    if data_choice == 1:
        train_set = datasets.MNIST(
            root="../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        test_set = datasets.MNIST(
            root="../data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_set = datasets.CIFAR10(
            root="../data", train=True, download=True, transform=transform
        )
        test_set = datasets.CIFAR10(
            root="../data", train=False, download=True, transform=transform
        )
        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    train_data = DataLoader(train_set)
    _, (example, _) = next(enumerate(train_data))
    data_params = {
        "input_dim": list(example[0].size()),
        "output_dim": len(classes),
        "fc": F.softmax,
    }

    # Defining model and moving to device
    out_message = """
Which model do you choose?
1: LeNet
2: Convolutional 2
3: Convolutional 4
4: Convolutional 6
Answer: """
    model_choice = 2
    while not (isinstance(model_choice, int)) or (model_choice not in [1, 2, 3, 4]):
        model_choice = str(input(out_message))
        try:
            model_choice = int(model_choice)
        except:
            model_choice = "p"

    if model_choice == 1:
        data_params["input_dim"] = list(torch.reshape(example[0], (-1,)).size())[0]
    else:
        data_params["n_channels"] = list(example[0].size())[0]

    # Get optimal parameters cited in the article
    optimal_params = get_optimal_hyperparams(model_choice)
    pruning_rate = optimal_params.pop("pruning_rate")

    # Set number of models for experimenting and getting average performance
    n_models = 5

    # Initializing loggers of performance metrics
    early_stopping_iter_list = list()
    train_acc_list = list()
    test_acc_list = list()

    # Iterating
    for _ in range(n_models):
        model = load_model(data_params, model_choice)
        model.set_device(device)
        # Training and monitoring the training
        remaining = 1
        pruning_level = list()
        early_stopping_iter = list()
        train_acc = list()
        test_acc = list()
        state_init = dict()
        prev_mod = dict()
        for key in model.state_dict().keys():
            state_init[key] = model.state_dict()[key].clone()

        if not os.path.exists(FIGURES_PATH):
            os.mkdir(FIGURES_PATH)

        while remaining > 0.05:

            _, validation_history = train_model(model, train_set, **optimal_params)
            train_acc.append(validation_history["accuracy"])
            early_stopping_iter.append(len(validation_history["accuracy"]))
            accur = 0
            for batch_index, (features, target) in enumerate(
                DataLoader(test_set, shuffle=True)
            ):
                features, target = features.to(device), target.to(device)
                accur += model.evaluate(features, target)
            accur /= len(test_set)
            test_acc.append(accur)
            print(accur)

            with torch.no_grad():
                model.prune(pruning_rate, state_init)
            pruning_level.append(remaining)
            remaining *= 1 - pruning_rate

        print(test_acc)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        early_stopping_iter_list.append(early_stopping_iter)

    train_acc_list = np.array(train_acc_list) * 100
    test_acc_list = np.array(test_acc_list) * 100
    early_stopping_iter_list = np.array(early_stopping_iter_list)

    # Changing to percentage
    pruning_level = [el * 100 for el in pruning_level]

    # Accuracy
    fig = plt.figure(figsize=(15, 15))
    plt.plot(pruning_level, np.mean(test_acc_list, axis=0), label="Mean")
    plt.plot(pruning_level, np.max(test_acc_list, axis=0), "--", label="Max")
    plt.plot(pruning_level, np.min(test_acc_list, axis=0), "--", label="Min")
    fig_name = (
        "Accuracy according to remaining weights - "
        + NETWORK_CHOICE[model_choice - 1]
        + " on "
        + DATA_CHOICES[data_choice - 1]
    )
    plt.title(fig_name)
    plt.xlabel("Remaining weights percentage")
    plt.ylabel("Accuracy in %")
    plt.xscale("log")
    plt.legend(loc="best")
    # plt.show()
    fig.savefig(os.path.join(FIGURES_PATH, fig_name + ".jpeg"))

    # Early stopping
    fig = plt.figure(figsize=(15, 15))
    plt.plot(pruning_level, np.mean(early_stopping_iter_list, axis=0), label="Mean")
    plt.plot(pruning_level, np.max(early_stopping_iter_list, axis=0), "--", label="Max")
    plt.plot(pruning_level, np.min(early_stopping_iter_list, axis=0), "--", label="Min")
    fig_name = (
        "Early stopping by remaining weights - "
        + NETWORK_CHOICE[model_choice - 1]
        + " on "
        + DATA_CHOICES[data_choice - 1]
    )
    plt.title(fig_name)
    plt.xlabel("Remaining weights percentage")
    plt.ylabel("Iteration")
    plt.xscale("log")
    plt.legend(loc="best")
    # plt.show()
    fig.savefig(os.path.join(FIGURES_PATH, fig_name + ".jpeg"))
