import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    N = X.shape[0]
    C = W.shape[1]

    for i in range(N):
        scores = X[i].dot(W)
        # Stabilize scores
        max_score = np.max(scores)
        scores -= max_score
        
        # Compute softmax probabilities
        exp_scores = np.exp(scores)
        sum_exp = np.sum(exp_scores)
        probs = exp_scores / sum_exp
        
        # Compute loss
        correct_class_prob = probs[y[i]]
        loss += -np.log(correct_class_prob)
        
        # Compute gradient
        for j in range(C):
            indicator = 1 if j == y[i] else 0
            dW[:, j] += (probs[j] - indicator) * X[i]

    # Average loss/gradient
    loss /= N
    dW /= N
    
    # Regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    
    Inputs:
    - W: Weights (D, C)
    - X: Data (N, D)
    - y: Labels (N,)
    - reg: Regularization strength
    
    Returns:
    - loss: Scalar
    - dW: Gradient (D, C)
    """
    num_train = X.shape[0]
    
    # Compute scores and stabilize
    scores = X.dot(W)
    max_scores = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    
    # Softmax probabilities
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Loss calculation
    correct_probs = probs[np.arange(num_train), y]
    loss = -np.sum(np.log(correct_probs)) / num_train
    loss += 0.5 * reg * np.sum(W * W)
    
    # Gradient calculation
    dscores = probs.copy()
    dscores[np.arange(num_train), y] -= 1
    dW = X.T.dot(dscores) / num_train
    dW += reg * W
    
    return loss, dW


