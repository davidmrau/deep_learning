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
from model import TextGenerationModel
def to_one_hot(indices, vocab_size, dtype):
    # init one hot
    zeros = torch.zeros(indices.shape[0],vocab_size).type(dtype)
    # fill one hot
    x_one_hot = zeros.scatter(1, indices.unsqueeze(-1), 1)
    return x_one_hot

def eval(config):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    # Initialize the device which to run the model on
    device = torch.device(device)
    dtype = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    # Initialize the dataset and data loader (note the +1)
    dataset = pickle.load(open(config.dataset_path, 'rb'))
    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, \
                 config.lstm_num_hidden, config.dropout_keep_prob, config.lstm_num_layers).to(device)

    model.load_state_dict(torch.load(config.model_path))
    # Setup the loss and optimizer
    model.eval()
    print('Evaluating: ')
    num_summaries = 5
    # get random intial chars
    rand_chars = [dataset._char_to_ix[random.choice(dataset._chars)] for i in range(num_summaries)]
    # to tensor
    prev_pred = torch.Tensor(rand_chars).type(dtype)
    prev_pred_one_hot = to_one_hot(prev_pred, dataset.vocab_size, dtype)
    predictions = []
    for i in range(config.sample_length):
        # batch size 1
        prev_pred_one_hot = torch.unsqueeze(prev_pred_one_hot, 1)
        if i is 0:
            y_pred, hidden = model(prev_pred_one_hot.float())
        else:
            y_pred, hidden = model(prev_pred_one_hot.float(), hidden)
        # get argmax
        # Sample from the network as a multinomial distribution
        if config.sampling_method == 'temp':
            output_dist = y_pred.data.div(config.temperature).exp()
            y_pred_batch_idx = output_dist.squeeze(1).multinomial(1).type(dtype)
        else:
            y_pred_batch_idx = y_pred.argmax(2).type(dtype)

        # to one hot
        prev_pred_one_hot = to_one_hot(y_pred_batch_idx.flatten(), dataset.vocab_size, dtype)
        predictions.append(y_pred_batch_idx.flatten().cpu().detach().numpy())
    predictions = np.asarray(predictions).T
    summaries = [dataset.convert_to_string(pred) for pred in list(predictions)]
    print("{} \n".format('\n'.join(summaries)))

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for temp sampling method')
    parser.add_argument('--sample_length', type=int, default=50, help='Temperature for temp sampling method')
    parser.add_argument('--sampling_method', type=str, default='temp', help="sampling method 'greedy' or 'temp'")
    parser.add_argument('--model_path', type=str, required=True, help="Path of the model")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    parser.add_argument('--dropout_keep_prob', type=float, default=0.6, help='Dropout keep probability')

    # Misc params
    parser.add_argument('--dataset_path', type=str,required=True, default="./models/", help='path for dataset')
    config = parser.parse_args()

    # Train the model
    eval(config)
