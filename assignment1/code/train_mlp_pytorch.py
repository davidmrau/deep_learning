"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch
import torch.nn as nn
import pickle

#import matplotlib.pyplot as plt


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 3500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.

  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch

  TODO:
  Implement accuracy computation.
  """

  accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1))/targets.shape[0]

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model.

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Device configuration
  use_cuda = torch.cuda.is_available()
  if use_cuda:
      print('Running in GPU model')

  device = torch.device('cuda' if use_cuda else 'cpu')
  dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

  # load dataset
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  # get batches
  batches = []

  # initializing loss and accuracy arrays
  accuracies = []
  losses = []

  for i in range(FLAGS.max_steps):
      x, y = cifar10['train'].next_batch(FLAGS.batch_size) # (batch_size, 3, 32, 32) (batch_size, 10)
      x = x.reshape(FLAGS.batch_size, -1)
      batches.append((x,y))
  # get output size
  out_size = batches[-1][1].shape[1]
  # get intput size
  in_size = batches[-1][0].shape[1]
  # initialize network
  net = MLP(in_size, dnn_hidden_units, out_size, FLAGS.batch_norm, FLAGS.dropout).to(device)


# initialize l1 regularization
  reg_factor = 1e-6

  # intialize loss function
  criterion = nn.CrossEntropyLoss()
  if FLAGS.l2:
      optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.learning_rate, weight_decay=1e-5)
  else:
      optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.learning_rate)
   # make steps
  for s in range(FLAGS.max_steps):
    net.train()
    x,t = batches[s]
    # Forward pass
    y = net(torch.from_numpy(x).type(dtype))
    t = torch.from_numpy(t).type(dtype)
    t = torch.max(t,1)[1]

    loss = criterion(y, t)

    if FLAGS.l1:
        l1_loss = torch.autograd.Variable(torch.FloatTensor(1), requires_grad=True)
        for name, param in net.named_parameters():
            if 'bias' not in name and isinstance(param, nn.Linear):
                loss = loss + (reg_factor * torch.sum(torch.abs(param)))
    losses.append(loss.cpu().detach().numpy())

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if s % FLAGS.eval_freq == 0:
        net.eval()
        x, t = cifar10['test'].images, cifar10['test'].labels
        x = x.reshape(x.shape[0],-1)
        y = net(torch.from_numpy(x).type(dtype))
        acc = accuracy(y.cpu().detach().numpy(),t)
        print('accuracy at step',s,': ',acc)
        print('loss at step',s,': ',loss.cpu().detach().numpy())
        accuracies.append(acc*100)

  save_acc = np.arange(len(accuracies)*FLAGS.eval_freq,step=FLAGS.eval_freq), accuracies
  save_losses = np.arange(len(losses)), losses
  acc_name = 'accuracies_lr_{}_l1_{}_batch_norm_{}_dropout_{}_h_{}'.format(FLAGS.learning_rate, FLAGS.l1, FLAGS.batch_norm,FLAGS.dropout, FLAGS.dnn_hidden_units)
  loss_name = 'losses_lr_{}_l1_{}_batch_norm_{}_dropout_{}_h_{}'.format(FLAGS.learning_rate, FLAGS.l1, FLAGS.batch_norm,FLAGS.dropout, FLAGS.dnn_hidden_units)
  pickle.dump(save_acc, open(acc_name.replace('.', '_')+'.p', 'wb'))
  pickle.dump(save_losses, open(loss_name.replace('.', '_')+'.p', 'wb'))

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--l2', action='store_true',
                    help='l2 loss')
  parser.add_argument('--l1', action='store_true',
                    help='l1 loss on linear layers')
  parser.add_argument('--batch_norm', action='store_true',
                    help='Batch normalization')
  parser.add_argument('--dropout', action='store_true',
                    help='Dropout 0.2')

  FLAGS, unparsed = parser.parse_known_args()

  main()
