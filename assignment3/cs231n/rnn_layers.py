from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    ht = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    #next_h = 2 / (1 + np.e ** (-2 * ht)) - 1
    next_h = np.tanh(ht)
    cache = (x, prev_h, Wx, Wh, b, ht)
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    x, prev_h, Wx, Wh, b, ht = cache

    #ht = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    #next_h = 2 / (1 + np.e ** (-2 * ht)) - 1
    #dht = 4 * (np.e ** (-2 * ht)) * dnext_h / ((1 + np.e ** (-2 * ht)) ** 2) #(N, H)
    dht = 4 * dnext_h / (2 + np.e ** (-2 * ht) + np.e ** (2 * ht))
    dx = np.dot(dht, Wx.T)
    dWx = np.dot(x.T, dht)
    dprev_h = np.dot(dht, Wh.T)
    dWh = np.dot(prev_h.T, dht)
    db = np.sum(dht, axis=0)

    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    N, T, D = x.shape
    _, H = Wx.shape
    cache = list()
    h = np.zeros((N, T, H))
    prev_h = h0
    for i in range(T):
        next_h, c = rnn_step_forward(x[:, i, :], prev_h, Wx, Wh, b)
        h[:, i, :] = next_h
        prev_h = next_h
        cache.append(c)
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    N, T, H = dh.shape
    x, prev_h, Wx, Wh, b, ht = cache[0]
    _, D = x.shape
    dx = np.zeros((N, T, D))
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)
    _dnext_h = np.zeros_like(prev_h)
    for i in range(T-1, -1, -1):
        # Note that at a step h generate h+1 and the output
        _dx, _dprev_h, _dWx, _dWh, _db = rnn_step_backward(dh[:, i, :] + _dnext_h, cache[i])
        dx[:, i, :] = _dx
        dWx += _dWx
        dWh += _dWh
        db += _db
        _dnext_h = _dprev_h
    dh0 = _dnext_h
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    # Word embedding traning input is one hot
    """
    N, T = x.shape
    V, D = W.shape
    x_one_hot = np.zeros((N, T, V))
    for i in range(N):
        for j in range(T):
            x_one_hot[i, j, x[i, j]] = 1
    out = np.dot(x_one_hot, W)
    """
    out = W[x, :]
    cache = (W, x)


    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    """
    W, x = cache
    N, T = x.shape
    V, D = W.shape
    x_one_hot = np.zeros((N, T, V))
    for i in range(N):
        for j in range(T):
            x_one_hot[i, j, x[i, j]] = 1
    #out = np.dot(x_one_hot, W)
    dW = np.dot(x_one_hot.transpose(2, 0, 1).reshape(V, -1), dout.reshape(-1, D))
    """
    # out = W[x, :]
    W, x = cache
    dW = np.zeros_like(W)  #(V, D)  dout: (N, T, D)  x(N, T, V)
    np.add.at(dW, x, dout)

    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

def tanh(x):
    return 2 * sigmoid(2 * x) - 1


def derivatives_sigmoid(x):
    """
    sigmoid

    f(z) = 1 / (1 + exp( − z))

    f(z)' = f(z)(1 − f(z))
    """
    return sigmoid(x) * (1 - sigmoid(x))

def derivatives_tanh(x):
    """
    tanh

    f(z) = tanh(z)

    f(z)' = 1 − (f(z))**2
    """
    return 1 - (tanh(x)) ** 2


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    N, H = prev_h.shape
    a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b  # (N, 4H)
    i = sigmoid(a[:, :H])
    f = sigmoid(a[:, H:2*H])
    o = sigmoid(a[:, 2*H:3*H])
    #g = 2 * sigmoid(2 * a[:, 3*H:]) - 1
    g = tanh(a[:, 3*H:])
    next_c = f * prev_c + i * g
    #next_h = o * (2 * sigmoid(2 * next_c) - 1)
    next_h = o * tanh(next_c)
    cache = x, prev_h, prev_c, Wx, Wh, b, next_h, next_c, g, o, f, i, a
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    x, prev_h, prev_c, Wx, Wh, b, next_h, next_c, g, o, f, i, a = cache
    # dnext_h, dnext_c
    N, H = dnext_h.shape
    da = np.zeros((N, 4*H))
    dnext_c += o * derivatives_tanh(next_c) * dnext_h   #(N,H)
    do = tanh(next_c) * dnext_h     # (N,H)
    df = prev_c * dnext_c           # (N,H)
    dprev_c = f * dnext_c           # (N,H)
    di = g * dnext_c                # (N,H)
    dg = i * dnext_c                # (N,H)
    da[:, 3*H:] = derivatives_tanh(a[:, 3*H:]) * dg             # (N,H)
    da[:, 2*H:3*H] = derivatives_sigmoid(a[:, 2*H:3*H]) * do    # (N,H)
    da[:, H:2*H] = derivatives_sigmoid(a[:, H:2*H]) * df        # (N,H)
    da[:, :H] = derivatives_sigmoid(a[:, :H]) * di              # (N,H)
    # da(N, 4H) Wx(D, 4H) x(N, D) Wh(H, 4H)
    dx = np.dot(da, Wx.T)   # (N,D)
    dWx = np.dot(x.T, da)   # (D,4H)
    dprev_h = np.dot(da, Wh.T)  # (N,H)
    dWh = np.dot(prev_h.T, da)  # (H,4H)
    db = np.sum(da, axis=0)
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    N, T, D = x.shape
    _, H = h0.shape
    h = np.zeros((N, T, H))
    next_c = np.zeros((N, H))
    cache = list()
    for i in range(T):
        h0, next_c, lstm_cache = lstm_step_forward(x[:, i, :], h0, next_c, Wx, Wh, b)
        h[:, i, :] = h0
        cache.append(lstm_cache)
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    N, T, H = dh.shape
    x, prev_h, prev_c, Wx, Wh, b, next_h, next_c, g, o, f, i, a = cache[0]
    _, D = x.shape
    dc = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dWh = np.zeros((H, 4*H))
    dWx = np.zeros((D, 4*H))
    db = np.zeros((4*H))
    dprev_h = np.zeros((N, H))
    for i in range(T-1, -1, -1):
        dx[:, i, :], dprev_h, dc, _dWx, _dWh, _db = lstm_step_backward(dh[:, i, :] + dprev_h, dc, cache[i])
        dWh += _dWh
        dWx += _dWx
        db += _db
    dh0 = dprev_h
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
