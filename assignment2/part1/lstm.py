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

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()

        self.device = device
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_hidden = num_hidden

        def init_param(shape):
            return nn.Parameter(torch.randn(shape, device=self.device, requires_grad=True ))

        self.Wgx = init_param((num_hidden, input_dim))
        self.Wgh = init_param((num_hidden, num_hidden))
        self.bg = init_param((num_hidden,1))
        self.Wix = init_param((num_hidden, input_dim))
        self.Wih = init_param((num_hidden, num_hidden))
        self.bi = init_param((num_hidden,1))
        self.Wfx = init_param((num_hidden, input_dim))
        self.Wfh = init_param((num_hidden, num_hidden))
        self.bf = init_param((num_hidden,1))
        self.Wox = init_param((num_hidden, input_dim))
        self.Woh = init_param((num_hidden, num_hidden))
        self.bo = init_param((num_hidden,1))
        self.Wph = init_param((num_classes, num_hidden))
        self.bp = init_param((num_classes,1))

    def forward(self, x):

        h_prev = torch.zeros(self.num_hidden, self.batch_size, device=self.device) # (h, b)
        c_prev = torch.zeros(self.num_hidden, self.batch_size, device=self.device)
        for t in range(x.shape[1]):
            x_t = x[:,t].view(1,-1)
            g = torch.tanh(self.Wgx @  x_t + self.Wgh @ h_prev + self.bg)
            i = nn.functional.sigmoid(self.Wix @  x_t + self.Wih @ h_prev + self.bi)
            f = nn.functional.sigmoid(self.Wfx @  x_t + self.Wfh @ h_prev + self.bf)
            o = nn.functional.sigmoid(self.Wox @  x_t + self.Woh @ h_prev + self.bo)
            c = g * i + c_prev * f
            h = torch.tanh(c) * o
            c_prev = c
            h_prev = h

        last_out = (self.Wph @ h + self.bp).transpose(1,0)
        return last_out
