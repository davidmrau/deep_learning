################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        modules = []
        # add hidden layers
        self.Whx = nn.Parameter(torch.randn((num_hidden, input_dim), device=device, requires_grad=True))
        self.Whh = nn.Parameter(torch.randn((num_hidden, num_hidden), device=device, requires_grad=True))
        self.bh = nn.Parameter(torch.randn((num_hidden,1), device=device, requires_grad=True))
        # add output layer
        self.Wph = nn.Parameter(torch.randn((num_classes, num_hidden), device=device, requires_grad=True) )
        self.bp = nn.Parameter(torch.randn((num_classes), device=device, requires_grad=True))
        self.seq_length = seq_length
        self.num_hidden = num_hidden
        self.batch_size = batch_size

    def forward(self, x):
        # Implementation here ...
        out = []
        #x # (b, i)
        h_prev = torch.zeros(self.num_hidden, self.batch_size)
        for t in range(x.shape[1]):
            # self.Whx @ x[:,t]: (h, i) (b,i)T = (h, b)
            # self.Whh @ h_prev: (h,h) (h,b) + (b) = (h,b)
            # h = (h,b)  (h,b) + (b)
            h = nn.functional.tanh(self.Whx @ x[:,t].view(1,-1) + self.Whh @ h_prev + self.bh)
            h_prev = h
        out =  h.transpose(1,0) @ self.Wph.transpose(1,0) + self.bp # (h, b)T (o, h)T  + (o) = (b, o)

        return out
