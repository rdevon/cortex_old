'''
Cost functions
'''

import theano
from theano import tensor as T


floatX = theano.config.floatX


def categorical_cross_entropy(y_hat, y, mask=None):
    """
    Inputs:
        y - 2D tensor (each row is a distribution) or vector of ints
            indicating the position in one-hot
        y_hat - 2D tensor (each row is a distribution)
    Outputs:
        cost - tensor of rank one less than y
    """
    cost = T.nnet.categorical_crossentropy(y_hat, y)
    if mask is not None:
        cost = cost * mask
    return cost.sum() / y_hat.shape[0]

def binary_cross_entropy(y_hat, y):
    cost = T.nnet.binary_crossentropy(y_hat, y)
    return cost.sum() / y_hat.shape[0]

def hinge_loss(y_hat, y):
    margins = 1 - y_hat * y + (y_hat * (1 - y)).max(1).dimshuffle(0, 'x')
    cost = T.mean(T.maximum(
        y_hat * T.constant(0.).astype(floatX), margins))
    return cost
