'''
Module for RBM tests.
'''

import theano
from theano import tensor as T

from cortex.datasets.basic import euclidean
from cortex.models import rbm
from cortex.utils import floatX


def test_build(dim_h=11, dim_v=13):
    model = rbm.RBM(dim_v, dim_h)
    model.set_tparams()
    return model

def test_sample(n_steps=3, dim_v=13, batch_size=7):
    data_iter = euclidean.Euclidean(dims=dim_v, batch_size=batch_size)
    x = data_iter.next()[data_iter.name]

    model = test_build(dim_v=dim_v)

    X = T.matrix('X', dtype=floatX)
    ph0 = model.ph_v(X)
    r = model.trng.uniform(size=(X.shape[0], model.dim_h))
    h_p = (r <= ph0).astype(floatX)

    outs, updates = model.sample(h_p, n_steps=n_steps)
    keys = outs.keys()

    f = theano.function([X], outs.values(), updates=updates)
    values = f(x)

    outs = model(X, n_chains=batch_size, n_steps=n_steps)
    results, samples, updates, constants = outs
    f = theano.function([X], results.values(), updates=updates)
    f(x)
