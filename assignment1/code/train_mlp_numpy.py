"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
from modules import LinearModule
import cifar10_utils

import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
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

  net = MLP(in_size, dnn_hidden_units, out_size)
  # intialize loss function
  criterion = CrossEntropyModule()

   # make steps
  for s in range(FLAGS.max_steps):
    x,t = batches[s]
     # forwardpass
    y = net.forward(x)
    # calculate loss
    loss = criterion.forward(y,t)
    losses.append(loss)
    # gradient for cross entropy
    dx = criterion.backward(y,t)
    # backward pass
    net.backward(dx)
    # update weights
    for m in net.modules:
        if isinstance(m, LinearModule):
            m.params['weight'] -= FLAGS.learning_rate * m.grads['weight']
            m.params['bias'] -= FLAGS.learning_rate * m.grads['bias']
    if s % FLAGS.eval_freq == 0:
        x, t = cifar10['test'].images, cifar10['test'].labels
        x = x.reshape(x.shape[0],-1)
        y = net.forward(x)
        acc = accuracy(y,t)
        print('accuracy at step',s,': ',acc)
        accuracies.append(acc*100)

  # plot accuracies and losses
  plt.subplot(2, 1, 1)
  plt.plot(np.arange(len(accuracies)*FLAGS.eval_freq,step=FLAGS.eval_freq), accuracies, 'o-')
  plt.title('Numpy MLP')
  plt.ylabel('Accuracy (%)')

  plt.subplot(2, 1, 2)
  plt.plot(np.arange(len(losses)), losses)
  plt.xlabel('Step')
  plt.ylabel('Cross Entropy Loss')

  plt.savefig('numpy_mlp.png')
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
  FLAGS, unparsed = parser.parse_known_args()

  main()
