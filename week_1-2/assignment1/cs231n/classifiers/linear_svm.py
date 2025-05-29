import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)  # (C,)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]         # Gradient update for incorrect class
                dW[:, y[i]] -= X[i]      # Gradient update for correct class

    # Average over number of training examples
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss and gradient
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_scores = scores[np.arange(num_train), y]
  margin = scores - correct_scores[:, np.newaxis] + 1
  margin[np.arange(num_train), y] = 0
  loss += np.sum(margin[margin>0])
  loss /= num_train
  loss += reg * np.sum(W * W)
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
  binary = (margin > 0).astype(float)  # (N, C)
  row_sum = np.sum(binary, axis=1)      # (N,)

  # For correct class, subtract row_sum for each sample
  binary[np.arange(num_train), y] = -row_sum

  # dW: X.T.dot(binary) gives the correct shape and semantics
  dW = X.T.dot(binary) / num_train
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
