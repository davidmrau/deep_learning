"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from custom_batchnorm import CustomBatchNormAutograd

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes, batch_norm, use_dropout):
    super(MLP, self).__init__()
    """
    Initializes MLP object.

    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP

    TODO:
    Implement initialization of the network.
    """
    self.modules = []
    # if list is empty -> no linear layer
    prev_size = n_inputs
    if len(n_hidden) > 0:
        for n in range(len(n_hidden)):
            self.modules.append(nn.Linear(prev_size, n_hidden[n]))
            prev_size = n_hidden[n]
    self.modules.append(nn.Linear(prev_size, n_classes))

    self.relu = nn.ReLU()
    self.layers = nn.ModuleList(self.modules)
    self.dropout = nn.Dropout(.2)
    self.use_dropout = use_dropout

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through
    several layer transformations.

    Args:
      x: input to the network
    Returns:
      out: outputs of the network

    TODO:
    Implement forward pass of the network.
    """

    out = x
    for i,m in enumerate(self.layers):
        # if not last layer
        if (i == len(self.layers) - 1):
            out = m(out)
        else:
            out = m(out)
            out = self.relu(out)
            if self.use_dropout:
                out = self.dropout(out)
    return out
