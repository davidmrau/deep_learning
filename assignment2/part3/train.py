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

import os
import time
from datetime import datetime
import argparse
import random
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# original
# TODO: Change this back
# from part3.dataset import TextDataset
# from part3.model import TextGenerationModel
# changed
from dataset import TextDataset
from model import TextGenerationModel
#

#

################################################################################

def train(config):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    # Initialize the device which to run the model on
    device = torch.device(device)
    dtype = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, \
                 config.lstm_num_hidden, config.lstm_num_layers, device).to(device)
    # Setup the loss and optimizer

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), config.learning_rate)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        model.train()
        # Only for time measurement of step through network
        t1 = time.time()


        y_pred_batch = model(batch_inputs.type(dtype))
        # get argmax
        y_pred_batch_idx = y_pred_batch.argmax(2)
        # initialize one hot
        y_pred_one_hot = torch.zeros_like(y_pred_batch)
        # copy indices into one hot with 1
        y_pred_one_hot = y_pred_one_hot.scatter(2, torch.unsqueeze(y_pred_batch_idx,2), 1).float()
        # merge batch and seq length
        y_pred_one_hot = y_pred_one_hot.view(-1,dataset.vocab_size)
        # merge batch and seq length
        batch_targets = batch_targets.view(-1)

        loss = criterion(torch.autograd.Variable(y_pred_one_hot, requires_grad=True), batch_targets.to(device))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        accuracy = np.sum(np.argmax(y_pred_one_hot.cpu().detach().numpy(), axis=1) == batch_targets.cpu().detach().numpy())/batch_targets.shape[0]
        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)
        config.train_steps = int(config.train_steps)
        if step % config.print_every == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                 "Accuracy = {:.2f}, Loss = {:.3f}".format(
                   datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                   config.train_steps, config.batch_size, examples_per_second,
                   accuracy, loss
            ))

        if step % config.sample_every == 0 :
            # Generate some sentences by sampling from the model
            model.eval()
            print('Evaluating: ')
            rand_chars = [dataset._char_to_ix[random.choice(dataset._chars)] for i in range(4)]
            prev_pred = torch.Tensor(rand_chars).type(dtype)
            prev_pred = prev_pred.unsqueeze(0)
            predictions = []
            for i in range(config.seq_length):
                y_pred = model(prev_pred.long())
                # get argmax
                y_pred_batch_idx = y_pred.argmax(2)
                prev_pred = y_pred_batch_idx
                predictions.append(y_pred_batch_idx.squeeze().cpu().detach().numpy())
            predictions = np.asarray(predictions).T
            print([dataset.convert_to_string(pred) for pred in list(predictions)])
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
