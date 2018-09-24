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
import pickle

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
################################################################################

def to_one_hot(indices, vocab_size):
    # init one hot
    zeros = torch.zeros(indices.shape[0],vocab_size)
    # fill one hot
    x_one_hot = zeros.scatter(1, indices.unsqueeze(-1), 1)
    return x_one_hot


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
                 config.lstm_num_hidden, config.dropout_keep_prob, config.lstm_num_layers).to(device)
    # Setup the loss and optimizer

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), config.learning_rate)

    lr = config.learning_rate
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        model.train()
        # Only for time measurement of step through network
        t1 = time.time()
        optimizer.zero_grad()
        # initialize one hot
        zeros = torch.zeros(batch_inputs.shape[0], config.seq_length ,dataset.vocab_size ).type(dtype)
        # create one hot
        batch_inputs_one_hot = zeros.scatter(2, batch_inputs.unsqueeze(-1),1)
        y_pred_batch, _ = model(batch_inputs_one_hot.type(dtype).float())
        # merge batch and seq length
        y_pred_one_hot = y_pred_batch.view(-1,dataset.vocab_size)
        # merge batch and seq length
        batch_targets = batch_targets.view(-1)
        #y_pred_one_hot = torch.autograd.Variable(y_pred_one_hot, requires_grad=True)
        loss = criterion(y_pred_one_hot, batch_targets.to(device))
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        loss.backward()
        optimizer.step()

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
            num_summaries = 5
            # get random intial chars
            rand_chars = [dataset._char_to_ix[random.choice(dataset._chars)] for i in range(num_summaries)]
            # to tensor
            prev_pred = torch.Tensor(rand_chars).type(dtype)
            prev_pred_one_hot = to_one_hot(prev_pred, dataset.vocab_size)
            predictions = []
            for i in range(config.seq_length):
                # batch size 1
                prev_pred_one_hot = torch.unsqueeze(prev_pred_one_hot, 0)
                if i is 0:
                    y_pred, hidden = model(prev_pred_one_hot)
                else:
                    y_pred, hidden = model(prev_pred_one_hot, hidden)
                # get argmax
                y_pred_batch_idx = y_pred.argmax(2)
                # to one hot
                prev_pred_one_hot = to_one_hot(y_pred_batch_idx.squeeze(), dataset.vocab_size)
                predictions.append(y_pred_batch_idx.squeeze().cpu().detach().numpy())
            predictions = np.asarray(predictions).T
            summaries = [dataset.convert_to_string(pred) for pred in list(predictions)]
            # write to file
            with open(config.summary_path+'summary.txt', 'a') as file:
                file.write("step {}: {} \n".format(step, '\t'.join(summaries)))
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break
        # adapt learning rate
        if step % config.learning_rate_step is 0 and step is not 0:
            lr = lr*config.learning_rate_decay
            print('Learning rate decreased: {} \n'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        # save model
        if step % config.save_every ==0:
            torch.save(model.state_dict(), config.save_path+'step_{}.pth'.format(step))
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
    parser.add_argument('--save_path', type=str, default="./models/", help='Output path for models')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=10, help='How often to sample from the model')
    parser.add_argument('--save_every', type=int, default=500, help='How often to save the model')

    config = parser.parse_args()

    # Train the model
    train(config)
