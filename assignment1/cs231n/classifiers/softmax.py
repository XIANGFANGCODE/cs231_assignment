import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  train_num = X.shape[0]
  class_num = W.shape[1]
  for i in range(train_num):
    scores = np.dot(X[i], W) # shape(C, )
    #lets make softmax function numeric stability
    max_scores = np.max(scores)
    scores -= max_scores
    correct_class_score = scores[y[i]]

    #forward
    expyi = np.e ** correct_class_score
    expj = np.e ** scores  # shape (C, )
    sum_expj = np.sum(expj)
    softmax = expyi / sum_expj
    logsoft = -np.log(softmax)
    loss += logsoft

    #backward   we need to different class j == y[i] and j != y[i]
    for j in range(class_num):
      #note that the shape of dW is shape (N,C)
      dsoftmax = -1 / softmax  # int
      dexpyi = (1 / sum_expj) * dsoftmax # int
      dsum_expj = (-expyi / (sum_expj ** 2)) * dsoftmax # int
      #the function sum's gradient is 1
      dexpj = dsum_expj
      dscores = (np.e ** scores[j]) * dexpj
      if j == y[i]:
        dcorrect_class_score = np.e ** scores[j] * dexpyi
      else:
        dcorrect_class_score = 0
      dW[:, j] += (dscores + dcorrect_class_score) * X[i].T

  loss /= train_num
  loss += reg * np.sum(W * W)

  dW /= train_num
  dW += 2 * reg * W

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive:
  
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  train_num = X.shape[0]
  class_num = W.shape[1]
  scores = np.dot(X, W)	 #shape (N,C)

  # make softmax function numeric stability
  # note that about argmax function, axis means which dimension 0 want to output
  max_scores_idx = np.argmax(scores, axis = 1)
  max_scores = scores[range(scores.shape[0]),max_scores_idx]
  scores = scores - max_scores.reshape(scores.shape[0], -1)
  correct_class_score = scores[range(train_num), y]	#shape (N, )

  # forward
  expyi = np.e ** correct_class_score #shape (N, )
  expj = np.e ** scores  #shape (N, C)
  sum_expj = np.sum(expj, axis = 1) #shape (N, )
  softmax = expyi / sum_expj #shape (N, )
  logsoft = -np.log(softmax) #shape (N, )
  loss = np.sum(logsoft)

  # backward
  # loss = np.sum(logsoft) this is sum the result inside the vector, so do not need compute the gradient?
  dsoftmax = -1 / softmax #shape (N, )
  dsum_expj = (-expyi / (sum_expj ** 2)) * dsoftmax #shape (N, )
  dexpyi = (1 / sum_expj) * dsoftmax #shape (N, )


  dexpj = np.ones_like(expj) * dsum_expj.reshape(expj.shape[0], -1) #shape (N,C)

  dscores = expj * dexpj #shape (N ,C)
  dcorrect_class_score = expyi * dexpyi #shape (N, )
  # X shape (N, D)
  dW = np.dot(X.T, dscores)  #shape (D,C)

  #for i in range(train_num):
  #  dW[:, y[i]] += dcorrect_class_score[i] * X[i].T

  #midv = dcorrect_class_score.reshape(train_num, -1) * X
  #for i in range(train_num):
  #  dW[:, y[i]] += midv[i]

  midv = np.zeros_like(dscores)
  midv[range(train_num), y] = dcorrect_class_score
  dW += np.dot(X.T, midv)


  # regularization
  loss /= train_num
  loss += reg * np.sum(W * W)
  dW /= train_num
  dW += 2 * reg * W

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

