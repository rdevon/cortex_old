'''
Tests for Local filter MLP
'''

import numpy as np
import theano
from theano import tensor as T

from cortex.models.mlp import LFMLP
from cortex.utils import floatX
from cortex.utils.tools import print_profile


def test_build(dim_f=3, prototype_shape=(10, 10, 10), shape=(3, 4, 5), stride=2,
               batch_size=3):
    prototype = np.random.randint(0, 2, size=prototype_shape).astype(floatX)
    print prototype.sum()
    model = LFMLP(prototype.sum(), 19, dim_f=3, prototype=prototype,
                  shape=shape, stride=stride)

    tparams = model.set_tparams()

    print_profile(tparams)

    X = T.matrix('X', dtype=floatX)
    outs = model(X)

    x = np.random.normal(size=(batch_size, prototype.sum())).astype(floatX)

    f = theano.function([X], outs.values())
    f(x)

    r_model = LFMLP(19, prototype.sum(), dim_f=3, prototype=prototype,
                    filter_in=False, shape=shape, stride=stride)

    tparams = r_model.set_tparams()

    print_profile(tparams)

    outs = r_model(X)

    f = theano.function([X], outs.values())

    x = np.random.normal(size=(batch_size, 19)).astype(floatX)
    f(x)