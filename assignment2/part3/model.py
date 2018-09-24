# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size, \
                 lstm_num_hidden=256, dropout_keep_prob=1, lstm_num_layers=2):

        super(TextGenerationModel, self).__init__()
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.batch_size = batch_size
        self.embedding = nn.Embedding(vocabulary_size,lstm_num_hidden)
        self.lstm = nn.LSTM(lstm_num_hidden, lstm_num_hidden, lstm_num_layers)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)
        self.dropout = nn.Dropout(dropout_keep_prob)

    def forward(self, x, h=None):
        embeds = self.embedding(x)
        if h:
            lstm_out, hidden = self.lstm(embeds, h)
        else:
            lstm_out, hidden = self.lstm(embeds)
        out = self.linear(lstm_out)
        out = self.dropout(out)
        return out
