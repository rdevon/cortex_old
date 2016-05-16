'''
Tests for VAE
'''

import numpy as np
import theano
from theano import tensor as T

from cortex.datasets.basic.euclidean import Euclidean
from cortex.inference import gdir
from cortex.models.helmholtz import Helmholtz
from cortex.utils import floatX
from cortex.utils.tools import print_profile, resolve_path


def test_build_GBN(dim_in=17, dim_h=13):
    rec_args = dict(input_layer='input')
    gen_args = dict(output='input')
    distributions = dict(input='gaussian')
    dims = dict(input=dim_in)
    gbn = Helmholtz.factory(dim_h, distributions=distributions, dims=dims,
                    rec_args=rec_args, gen_args=gen_args)
    tparams = gbn.set_tparams()

    print_profile(tparams)

    return gbn

def test_call():
    data_iter = Euclidean(batch_size=27, dim_in=17)
    gbn = test_build_GBN(dim_in=data_iter.dims[data_iter.name])

    X = T.matrix('x', dtype=floatX)
    results, samples, _, _ = gbn(X, X, n_posterior_samples=7)

    f = theano.function([X], samples.values() + results.values())

    x = data_iter.next()[data_iter.name]
    f(x)