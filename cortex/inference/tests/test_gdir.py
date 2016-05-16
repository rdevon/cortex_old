'''
Test GDIR
'''

import numpy as np
import theano
from theano import tensor as T

from cortex.datasets.basic.euclidean import Euclidean
from cortex.inference.gdir import MomentumGDIR
from cortex.models.tests import test_vae
from cortex.utils import floatX
from cortex.utils.tools import resolve_path

def test_build_gdir(model=None, **inference_args):
    if model is None:
        data_iter = Euclidean(batch_size=27, dim_in=17)
        model = test_vae.test_build_GBN(dim_in=data_iter.dims[data_iter.name])
    gdir = MomentumGDIR(model, **inference_args)
    return gdir

def test_infer():
    data_iter = Euclidean(batch_size=27, dim_in=17)
    gbn = test_vae.test_build_GBN(dim_in=data_iter.dims[data_iter.name])

    inference_args = dict(
        n_inference_steps=7,
        pass_gradients=True
    )

    gdir = test_build_gdir(gbn, **inference_args)

    X = T.matrix('x', dtype=floatX)

    rval, constants, updates = gdir.inference(X, X)

    f = theano.function([X], rval.values(), updates=updates)
    x = data_iter.next()[data_iter.name]

    results, samples, full_results, updates = gdir(X, X)
    f = theano.function([X], results.values(), updates=updates)

    print f(x)
