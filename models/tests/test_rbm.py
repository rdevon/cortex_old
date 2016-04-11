'''
Module for RBM tests.
'''

import theano
from theano import tensor as T

from datasets import euclidean
from models import rbm
from utils import floatX


def test_build(dim_h=11, dim_v=13):
    model = rbm.RBM(dim_v, dim_h)
    model.set_tparams()
    return model

def test_sample(n_steps=3, n_samples=5, dim_v=13):
    data_iter = euclidean.Euclidean(dims=dim_v, batch_size=7)
    x = data_iter.next()[data_iter.name]

    model = test_build(dim_v=dim_v)

    X = T.matrix('X', dtype=floatX)
    ph0 = model.step_ph_v(X, model.W, model.c)
    r = model.trng.uniform(size=(n_samples, X.shape[0], model.dim_h))
    h_p = (r <= ph0).astype(floatX)
    outs, updates = model.sample(h_p, n_steps=n_steps)
    keys = outs.keys()

    f = theano.function([X], outs.values(), updates=updates)

    values = f(x)
    '''
    for k, v in zip(keys, values):
        print k
        print v
    '''
    '''
    os = model(X, n_chains=n_samples, n_steps=n_steps)
    f = theano.function([X], os, updates=updates)
    outs = f(x)
    assert False, [o.shape for o in outs]
    '''
    outs = model(X, n_chains=n_samples, n_steps=n_steps)
    results, samples, updates, constants = outs
    f = theano.function([X], results.values(), updates=updates)
    assert False, f(x)
