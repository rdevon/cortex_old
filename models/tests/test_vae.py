'''
Tests for VAE
'''

import numpy as np
import theano
from theano import tensor as T

from datasets.mnist import MNIST
from inference import gdir
from models.gbn import GBN
from utils.tools import (
    floatX,
    print_profile
)


def test_build_GBN(dim_in=17, dim_h=13):
    gbn = GBN(dim_in, dim_h)
    tparams = gbn.set_tparams()

    print_profile(tparams)

    return gbn

def test_call():
    data_iter = MNIST(source='/Users/devon/Data/mnist.pkl.gz', batch_size=27)
    gbn = test_build_GBN(dim_in=data_iter.dims[data_iter.name])

    X = T.matrix('x', dtype=floatX)
    results, samples = gbn(X, X, n_samples=7)

    f = theano.function([X], samples.values() + results.values())

    x = data_iter.next()[data_iter.name]
    assert False, f(x)