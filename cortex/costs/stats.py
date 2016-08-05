'''Basic stats

'''

from collections import OrderedDict
from theano import tensor as T


def logistic_regression_stats(P=None, Y=None):
    stats = OrderedDict()
    Y_pred = T.argmax(P, axis=-1)
    Y_ = T.argmax(Y, axis=-1)
    stats['logistic_regression.error'] = T.neq(Y_pred, Y_).mean()
    stats['logistic_regression.nll'] = (
        -Y * T.log(P) - (1 - Y) * T.log(1 - P)).sum(axis=-1).mean()
    return stats

_stats = {'logistic_regression': logistic_regression_stats}