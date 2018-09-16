"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
import torch.nn as nn
import pickle


# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
  Performs training and evaluation of ConvNet model.

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42) 
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
      batches.append((x,y))
  # get output size
  out_size = batches[-1][1].shape[1]
  # get intput size
  in_size = batches[-1][0].shape[1]
  # initialize network
  net = ConvNet(in_size, out_size).to(device)
  net = nn.DataParallel(net)
  # intialize loss function
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.learning_rate)
   # make steps
  for s in range(FLAGS.max_steps):
    x,t = batches[s]
    # Forward pass
    y = net(torch.from_numpy(x).type(dtype))
    t = torch.from_numpy(t).type(dtype)
    t = torch.max(t,1)[1]
    loss = criterion(y, t)
    losses.append(loss.cpu().detach().numpy())


    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if s % FLAGS.eval_freq == 0:
        x, t = cifar10['test'].images, cifar10['test'].labels
        y = net(torch.from_numpy(x).type(dtype))
        acc = accuracy(y.cpu().detach().numpy(),t)
        accuracies.append(acc)
        print('accuracy at step',s,': ',acc)

  save_acc = np.arange(len(accuracies)*FLAGS.eval_freq,step=FLAGS.eval_freq), accuracies
  save_losses = np.arange(len(losses)), losses
  pickle.dump(save_acc, open('accuracies.p', 'wb'))
  pickle.dump(save_losses, open('losses.p', 'wb'))
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
  FLAGS, unparsed = parser.parse_known_args()

  main()
