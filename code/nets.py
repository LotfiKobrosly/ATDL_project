import os

import numpy as np
import pandas as pd

import torch
from torch import nn, optim
import torch.nn.functional as F

class NeuralNet(nn.Module):
    """
    """

    def __init__(self,
        dim_input,
        dim_output,
        n_hidden_layers,
        n_hidden_units,
        loss):
        super().__init__()
        #self.input = nn.Input(dim_input)
        self.dims = [dim_input] + n_hidden_units + [dim_output]
        self.layers = [nn.Linear(self.dims[i], self.dims[i+1]) for i in range(n_hidden_layers+1)]


    def forward(self, X):
        """
        """
        for i, layer in enumerate(self.layers[:-1]):
            X = F.relu(layer(X))
        return F.softmax(X)

    def train(self, X, y, batch_size, lr):
        """
        """
        pass

    def predict(self, X):
        """
        """
        pass

    def evaluate(self, X, y):
        """
        """
        y_pred = self.predict(X)
        return self.loss(y, y_pred)


    def prune(self, percentage):
        """
        """
        pass

    def evaluate_performance(self,
        X_train,
        y_train,
        X_test,
        y_test,
        n_epochs,
        batch_size,
        lr):
        """
        """

        # Initializing performance measures
        train_accuracy = list()
        test_accuracy = list()
        for i in range(n_epochs):
            train_accuracy.append(self.train(
                X_train,
                y_train,
                batch_size=batch_size,
                lr=lr)
            test_accuracy.append(self.evaluate(X_test, y_test))

