import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive_numerically(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero 

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
      

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  # Computing the gradient numerically with finite differences
  h = 0.00001
  # iterate over all indexes in W
  it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
  decW = W
  while not it.finished:
    # evaluate function at x+h
    ix = it.multi_index
    old_value = W[ix]
    W[ix] = old_value + h # increment by h
    decW[ix] = old_value -h # decrease by h 
    fx_inc_h = 0.0
    fx_dec_h = 0.0
    for i in xrange(num_train):
      scores = X[i].dot(W)
      dec_scores = X[i].dot(decW)
      correct_class_score = scores[y[i]]
      dec_correct_class_score = dec_scores[y[i]]
      for j in xrange(num_classes):
        if j == y[i]:
          continue
        margin = scores[j] - correct_class_score + 1 # note delta = 1
        if margin > 0:
          fx_inc_h += margin
        margin = dec_scores[j] - dec_correct_class_score + 1
        if margin > 0:
          fx_dec_h += margin
    fx_inc_h /= num_train
    fx_inc_h += reg * np.sum(W * W)
    fx_dec_h /= num_train
    fx_dec_h += reg * np.sum(decW * decW) 
 
    # restore to previous value (very important!)
    W[ix] = old_value 
    decW[ix] = old_value 
    
    # compute the partial derivative
    dW[ix] = (fx_inc_h - fx_dec_h) / (2 * h) # the slope
    it.iternext() # step to next dimension  
  
  return loss, dW

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero 

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i].T
        dW[:,y[i]] -= X[i].T  

  dW /= num_train
  dW += 2 * reg * W
             

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

        
        
        
    

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.

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
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W) # N*C 
  # create a correct_class_score_vector
  correct_class_score = scores[range(num_train), y]

  #note that 
  #scores -=  correct_class_score.reshape(num_train, -1) + 1 is wrong
  scores = scores - correct_class_score.reshape(num_train, -1) + 1
  scores[scores < 0] = 0
  # if j = yj, set 0 to loss
  scores[range(num_train), y] = 0
  loss = np.sum(scores)
  loss /= num_train
  loss += reg * np.sum(W * W)  
  
  # compute dW(D*C)  X(N*D)
  scores[scores > 0] = 1
  scores_sum_by_column = np.sum(scores, axis = 1) # (C,)
  scores[range(num_train), y] = -scores_sum_by_column  #note the - value
  
  #X_sum_by_row = np.sum(X, axis = 0)  # (D,)
  dW = np.dot(X.T, scores)
  dW /= num_train
  dW += 2 * reg * W

  

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
