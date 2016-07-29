'''Basic stats

'''

from collections import OrderedDict
from theano import tensor as T


def logistic_regression_stats(P=None, Y_hat=None):
    stats = OrderedDict()
    Y = T.argmax(P, axis=-1)
    Y_ = T.argmax(Y_hat, axis=-1)
    stats['logistic_regression.error'] = T.neq(Y, Y_).mean()
    stats['logistic_regression.nll'] = (-Y_hat * T.log(P) - (1 - Y_hat) * T.log(1 - P)).sum(axis=-1).mean()

    return stats

_stats = {'logistic_regression': logistic_regression_stats}