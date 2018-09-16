"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data.
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module.

    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and
    std = 0.0001. Initialize biases self.params['bias'] with 0.

    Also, initialize gradients with zeros.
    """


    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}

    self.activation = None # # NOTE: should be size out_features
    self.params['weight'] = np.random.normal(loc=0.0, scale=0.0001, size=(in_features, out_features))
    self.params['bias'] = np.zeros((out_features,))
    self.grads['weight'] = np.zeros((in_features, out_features))
    self.grads['bias'] = np.zeros((out_features,))

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    out = x @ self.params['weight'] + self.params['bias'] # (b,out) = (b, in)(in, out)
    self.activation = x
    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    Implement backward pass of the module. Store gradient of the loss with respect to
    layer parameters in self.grads['weight'] and self.grads['bias'].
    """
    self.grads['weight'] = self.activation.T @ dout # (in,out) = (in, b)(b, out)
    self.grads['bias'] = np.sum(dout,axis=0) # (out)
    dx = dout @  self.params['weight'].T # (b, in) = (b, out)(out, in)
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """

  def __init__(self):
      self.activation = None

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    out = np.maximum(x, 0)
    self.activation = x
    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    dx = dout * (self.activation > 0).astype(int)
    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """

  def __init__(self):
      self.activation = None


  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    b = x.max(axis=1, keepdims=True)
    y = np.exp(x - b)
    out = y / y.sum(axis=1, keepdims=True)
    self.activation = out

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    dout = np.expand_dims(dout, axis=2)
    dots = np.einsum('ij, ik -> ijk', self.activation, self.activation)
    diagonals = np.apply_along_axis(np.diag, axis=1, arr=self.activation)
    gradient = diagonals - dots
    dx= np.einsum('ijk, ikl -> ijl', gradient, dout)
    dx = dx.squeeze(axis=2)
    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """

  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    TODO:
    Implement forward pass of the module.
    """
    ## NOTE: implemented for binary targets
    out = -np.sum(np.log(np.diag(x[:,y.argmax(1)])))/y.shape[0]
    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """
    dx = np.zeros((y.shape))
    for i,j in enumerate(y.argmax(1)):
        dx[i,j] = (-1/x[i,j])/y.shape[0]
    return dx
