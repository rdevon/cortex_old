'''Extra ops not implemented in Theano.

'''

import numpy as np
from numpy.linalg import slogdet
from theano.gof import Apply, Op
from theano import tensor as T
from theano.tensor.nlinalg import matrix_inverse


class LogAbsDet(Op):
    '''Log abs determinant of a matrix.

    Numerically stable version that relies on numpy.linalg.slogdet.

    '''
    __props__ = ()

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        assert x.ndim == 2
        o = T.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])

    def perform(self, nose, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        try:
            z[0] = np.asarray(slogdet(x)[1], dtype=x.dtype)
        except Exception:
            print('Failed to compute log abs det', x)
            raise

    def grad(self, inputs, g_outputs):
        [gz] = g_outputs
        [x] = inputs
        return [gz * matrix_inverse(x).T]

    def infer_shape(self, node, shapes):
        return [()]

    def __str__(self):
        return 'LogAbsDet'
logabsdet = LogAbsDet()
