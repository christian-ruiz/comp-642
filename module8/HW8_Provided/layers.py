import numpy as np


def affine_forward(x, theta, theta0):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (m, d_1, ..., d_k) and contains a minibatch of m
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension d = d_1 * ... * d_k, and
  then transform it to an output vector of dimension h.

  Inputs:
  - x: A numpy array containing input data, of shape (m, d_1, ..., d_k)
  - theta: A numpy array of weights, of shape (d, h)
  - theta0: A numpy array of biases, of shape (h,)
  
  Returns a tuple of:
  - out: output, of shape (m, h)
  - cache: (x, theta, theta0)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  # 2 lines of code expected
  shape_x = x.shape
  x = np.reshape(x, newshape=(x.shape[0], theta.shape[0]))
  out = np.dot(x, theta) + np.repeat([theta0],[x.shape[0]],axis = 0)
  x = np.reshape(x, newshape=shape_x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, theta, theta0)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (m, h)
  - cache: Tuple of:
    - x: Input data, of shape (m, d_1, ... d_k)
    - theta: Weights, of shape (d,h)
    - theta0: biases, of shape (h,)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (m, d1, ..., d_k)
  - dtheta: Gradient with respect to theta, of shape (d, h)
  - dtheta0: Gradient with respect to theta0, of shape (h,)
  """
  x, theta, theta0 = cache
  dx, dtheta, dtheta0 = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  # Hint: do not forget to reshape x into (m,d) form
  shape_x = x.shape
  x = np.reshape(x,newshape=(x.shape[0],theta.shape[0]))
  # 4-5 lines of code expected
  dtheta0 = np.sum(dout, axis=0)
  dtheta = np.dot(np.transpose(x),dout)
  dx = np.dot(dout, np.transpose(theta))
  dx = np.reshape(dx,shape_x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dtheta, dtheta0


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  # 1 line of code expected
  out = np.maximum(x, 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  # 1 line of code expected. Hint: use np.where
  # newx: m,d
  dout[np.where(x<=0)] = 0
  dx = dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    # 2 lines of code expected
    mask = np.random.binomial(1, p, size=x.shape)
    out = x * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    # 1 line of code expected
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    # 1 line of code expected
    dx = np.where(mask > 0.0, dout, 0.0)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, theta, theta0, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of m data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (m, C, H, W)
  - theta: Filter weights of shape (F, C, HH, WW)
  - theta0: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (m, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, theta, theta0, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################

  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  H = x.shape[2]
  W = x.shape[3]
  pad_x = np.pad(x, pad_width=((0,0),(0,0),(conv_param['pad'],conv_param['pad']),(conv_param['pad'],conv_param['pad'])), mode='constant', constant_values=0)

  HH = theta.shape[2]
  WW = theta.shape[3]

  H_act = 1 + (H + 2 * conv_param['pad'] - HH) / conv_param['stride']
  W_act = 1 + (W + 2 * conv_param['pad'] - WW) / conv_param['stride']

  out = np.zeros((x.shape[0],theta.shape[0],H_act, W_act))
  for i in range(len(pad_x)):
    # for each image calculate
    input_ = pad_x[i]
    for j in range(len(theta)):
      kernel_ = theta[j]
      for _h_act in range(H_act):
        start_h = _h_act*conv_param['stride']
        for _w_act in range(W_act):
          start_w = _w_act*conv_param['stride']
          region = input_[:,start_h:start_h+HH,start_w:start_w+WW]
          # region and kernel_ should have the same size
          act_value = np.sum(region*kernel_)+theta0[j]
          out[i,j,_h_act,_w_act] = act_value
  #out = np.reshape(out, newshape=(1, out.shape[0], out.shape[1], out.shape[2], out.shape[3]))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, theta, theta0, conv_param)

  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, theta, theta0, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dtheta: Gradient with respect to theta
  - dtheta0: Gradient with respect to theta0
  """
  dx, dtheta, dtheta0 = None, None, None

  x, theta, theta0, conv_param = cache
  #(F, C, HH, WW)
  dtheta = np.zeros(theta.shape)
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  m, C, H, W = x.shape
  F, C, HH, WW = theta.shape
  stride = conv_param['stride']
  pad = conv_param['pad']
  H_act = 1 + (H + 2 * pad - HH) / stride
  W_act = 1 + (W + 2 * pad - WW) / stride
  pad_x = np.pad(x, pad_width=((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant', constant_values=0)
  dx = np.zeros(pad_x.shape)
  dtheta0 = np.zeros(theta0.shape)
  for i in range(m):
    # the ith image
    input_ = pad_x[i]
    for j in range(F):
      # the jth kernel
      kernel = theta[j]
      for _h_act in range(H_act):
        start_h = _h_act*stride
        for _w_act in range(W_act):
          start_w = _w_act*stride
          dout_pos = np.ones(dtheta[j].shape)*dout[i, j, _h_act, _w_act]
          dtheta[j] += input_[:,start_h:start_h+HH, start_w:start_w+WW]*dout_pos
          dx[i,:,start_h:start_h+HH, start_w:start_w+WW] += theta[j]*dout_pos
      dtheta0[j] += np.sum(dout[i, j, :, :])
  dx = dx[:,:,pad:pad_x.shape[2]-pad,pad:pad_x.shape[3]-pad]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dtheta, dtheta0


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (m, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None

  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  m, C, H, W = x.shape
  stride = pool_param['stride']
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  new_w = 1 + (W - pool_width) / stride
  new_h = 1 + (H - pool_height) / stride
  out = np.zeros(shape=(m, C, new_h, new_w))
  pool_param['index'] = []

  for i in range(m):
    input_ = x[i]
    for j in range(C):
      for k in range(new_h):
        start_h = k * stride
        for l in range(new_w):
          start_w = l * stride
          out[i,j,k,l] = np.max(input_[j,start_h:start_h+pool_height,start_w:start_w+pool_width])
          #print(np.where(input_ ==out[i,j,k,l]))
          pool_param['index'].append(np.where(x ==out[i,j,k,l]))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  m, C, H, W = x.shape
  dx = np.zeros(x.shape)
  index_list = pool_param['index']
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  count = 0
  for i in range(m):
    for j in range(C):
      for k in range(dout.shape[2]):
        for l in range(dout.shape[3]):
          index = index_list[count]
          dx[index] = dout[i,j,k,l]
          count+=1
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx



def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (m, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (m,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  m = x.shape[0]
  correct_class_scores = x[np.arange(m), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(m), y] = 0
  loss = np.sum(margins) / m
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(m), y] -= num_pos
  dx /= m
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (m, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (m,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  m = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(m), y])) / m
  dx = probs.copy()
  dx[np.arange(m), y] -= 1
  dx /= m
  return loss, dx
