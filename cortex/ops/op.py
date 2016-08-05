'''Ops

'''

import numpy as np
import theano
from theano import tensor as T

from ..utils import floatX


class Op(object):
    def __init__(self, op, *args):
        self.dim_in = None
        self.dim_out = None
        self.op = op
        self.args = args

    def set_link_value(self, dim_in):
        X = T.vector('op_test', dtype=floatX)
        f = theano.function([X], self.op(X))
        x = np.ones((dim_in,)).astype(floatX)
        self.dim_out = f(x).shape[0]

    def get_link_value(self, link):
        if self.dim_out is None:
            raise KeyError
        return self.dim_out

