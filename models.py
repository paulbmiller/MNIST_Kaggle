# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from pytorch_utils import NeuralNet


class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of
    networks. This makes it easier to navigate the network with introspection.
    """

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class CNN_basic(nn.Module):
    """
    Convolutional neural network with 3 layers and a fully connected linear
    layer.

    Attributes
    ----------
    conv1 : torch.nn.Sequential
        1st Convolutional layer
    conv2 : torch.nn.Sequential
        2nd Convolutional layer
    mp : torch.nn.Sequential
        Max-Pool layer
    fc : torch.nn.Sequential
        Final classification fully connected layer
    """

    def __init__(self, lr, in_channels=1, out_channels=10,
                 optimizer=torch.optim.Adam):
        """
        Creates the CNN_basic model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        input_channels : int
            Dimensionality of the input
        """
        super(CNN_basic, self).__init__()
        self.model = NeuralNet.Net()
        self.model.add_conv_layer(in_channels, 4, kernel_size=4,
                                  stride=2)
        self.model.add_leakyrelu()
        self.model.add_conv_layer(4, 16, kernel_size=3, stride=2)
        self.model.add_leakyrelu()
        self.model.add_maxpool(kernel_size=4, stride=1)
        self.model.add_leakyrelu()
        self.model.add_layer(Flatten())
        self.model.add_linear(16*3*3, out_channels)
        self.model.add_sigmoid()
        self.model.init_optim(optimizer, lr)

    def forward(self, x):
        """
        Computes forward pass on the network

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """
        return self.model(x)
