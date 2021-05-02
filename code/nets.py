import os

from abc import abstractmethod

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune


def reinit_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class GeneralNet(nn.Module):

    def prune(self, percentage, state_init):
        """

        def add_masks(model, masks):
            mask_pruner = prune.CustomFromMask(None)
            for module_name, module in model.named_modules():
                key = f"{module_name}.weight_mask"
                if key in masks:
                    if isinstance(module, torch.nn.Conv2d):
                        _mask = masks[key]
                        mask_pruner.apply(module, 'weight', _mask)
                    if isinstance(module, torch.nn.Linear):
                        _mask = masks[key]
                        mask_pruner.apply(module, 'weight', _mask)

        def merge_masks(model):
            for n, m in model.named_modules():
                if not prune.is_pruned(m):
                    continue
                if isinstance(m, torch.nn.Conv2d):
                    prune.remove(m, name='weight')
                if isinstance(m, torch.nn.Linear):
                    prune.remove(m, name='weight')"""
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    self.layers[i] = prune.ln_structured(
                        layer,
                        "weight",
                        amount=percentage,
                        dim=0,
                        n=2
                    )
                    for name, buf in layer.named_buffers():
                        if name == "weight_mask":
                            break
                    print(buf)
                    self.layers[i].weight = buf * state_init["layers."+str(i)+".weight"]
                    print(self.layers[i].weight)

    @abstractmethod
    def forward(self, X):
        pass

    def evaluate(self, X, y):
        """
        """
        y_pred = self.forward(X)
        _, y_pred = torch.max(y_pred, 1)
        return (y == y_pred).sum().item() / list(X.size())[0]


class NeuralNet(GeneralNet):
    """
    """

    def __init__(self,
                 input_dim=100,
                 output_dim=200,
                 n_hidden_layers=100,
                 n_hidden_units=[],
                 fc=None):
        super(NeuralNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dims = [input_dim] + n_hidden_units + [output_dim]
        self.layers = nn.ModuleList([nn.Flatten()])
        self.layers.extend([nn.Linear(self.dims[i], self.dims[i+1]) for i in range(n_hidden_layers + 1)])
        self.fc = fc
        self.device = "cpu" # Default

    def set_device(self, device):
        self.device = device
        self.to(self.device)

    def forward(self, x):
        """
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        return self.fc(x)


class ConvNet(GeneralNet):
    """
    """

    def __init__(self,
                 input_dim=1,
                 output_dim=1,
                 n_channels=1,
                 n_conv_steps=1,
                 n_dense=1,
                 fc=None):
        super(ConvNet, self).__init__()
        self.conv_layers = list()
        new_input_h, new_input_w = input_dim[1], input_dim[2]
        for _ in range(n_conv_steps):
            self.conv_layers.extend([
                nn.Conv2d(in_channels=n_channels,
                          out_channels=n_channels,
                          kernel_size=(3, 3)),
                nn.Conv2d(in_channels=n_channels,
                          out_channels=n_channels,
                          kernel_size=(3, 3)),
                nn.MaxPool2d(kernel_size=(2, 2))
            ])
            new_input_w = (new_input_w - 4) // 2
            new_input_h = (new_input_h - 4) // 2
        self.flatten = nn.Flatten()
        self.dense_layers = []
        print("New dims: ", new_input_w, " and ", new_input_h)
        dims = [new_input_h*new_input_w*n_channels] + [256] * n_dense
        for i in range(n_dense):
            self.dense_layers.append(nn.Linear(dims[i], dims[i+1]))

        self.out = nn.Linear(256, output_dim)
        self.layers = nn.ModuleList(self.conv_layers + [self.flatten] + self.dense_layers + [self.out])
        self.fc = fc
        self.device = "cpu"

    def set_device(self, device):
        self.device = device
        self.to(self.device)

    def forward(self, x):
        """
        """
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        for layer in self.dense_layers:
            x = F.relu(layer(x))
        x = self.fc(self.out(x))
        return x
