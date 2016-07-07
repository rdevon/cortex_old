'''
Tests for DARN model.
'''

import numpy as np
import theano
from theano import tensor as T

from cortex.models.darn import (
    AutoRegressor,
    DARN
)
from cortex.models.helmholtz import Helmholtz
from cortex.utils import floatX


sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

def test_autoregressor(dim=3, n_samples=5):
    ar = AutoRegressor(dim)
    ar.params['b'] += 0.1
    tparams = ar.set_tparams()

    X = T.matrix('X', dtype=floatX)
    nlp = ar.neg_log_prob(X)
    p = ar.get_prob(X, *ar.get_params())
    W = T.tril(ar.W, k=-1)
    z = T.dot(X, W) + ar.b

    x = np.random.randint(0, 2, size=(n_samples, dim)).astype(floatX)

    f = theano.function([X], [nlp, p, z, W])
    nlp_t, p_t, z_t, W_t = f(x)
    print x.shape, nlp_t.shape
    z_np = np.zeros((n_samples, dim)).astype(floatX) + ar.params['b'][None, :]

    for i in xrange(dim):
        print i
        for j in xrange(i + 1, dim):
            print i, j
            z_np[:, i] += ar.params['W'][j, i] * x[:, j]

    assert np.allclose(z_t, z_np), (z_t, z_np)
    p_np = sigmoid(z_np)
    assert np.allclose(p_t, p_np, atol=1e-4), (p_t - p_np)

    p_np = np.clip(p_np, 1e-7, 1 - 1e-7)
    nlp_np = (- x * np.log(p_np) - (1 - x) * np.log(1 - p_np)).sum(axis=1)

    assert np.allclose(nlp_t, nlp_np, atol=1e-3), (nlp_t - nlp_np)

    samples, updates = ar.sample(n_samples=n_samples)

    f = theano.function([], samples, updates=updates)
    print f()

def test_darn(dim_in=5, dim_h=3, dim_out=7, n_samples=13):
    darn = DARN(dim_in, dim_h, dim_out, 2, h_act='T.tanh', out_act='T.nnet.sigmoid')
    tparams = darn.set_tparams()

    X = T.matrix('X', dtype=floatX)
    H = T.matrix('H', dtype=floatX)
    C = darn(H)
    NLP = darn.neg_log_prob(X, C)

    f = theano.function([X, H], [C, NLP])

    x = np.random.randint(0, 2, size=(n_samples, dim_out)).astype(floatX)
    h = np.random.randint(0, 2, size=(n_samples, dim_in)).astype(floatX)

    c_t, nlp_t = f(x, h)
    print c_t.shape

    d_np = np.tanh(np.dot(h, darn.params['W0']) + darn.params['b0'])
    c_np = np.dot(d_np, darn.params['W1']) + darn.params['b1']

    assert np.allclose(c_t, c_np), (c_t, c_np)

    z_np = np.zeros((n_samples, dim_out)).astype(floatX) + darn.params['bar'][None, :] + c_np

    for i in xrange(dim_out):
        for j in xrange(i + 1, dim_out):
            z_np[:, i] += darn.params['War'][j, i] * x[:, j]

    p_np = sigmoid(z_np)

    p_np = np.clip(p_np, 1e-7, 1 - 1e-7)
    nlp_np = (- x * np.log(p_np) - (1 - x) * np.log(1 - p_np)).sum(axis=1)

    assert np.allclose(nlp_t, nlp_np), (nlp_t, nlp_np)

    samples, updates_s = darn.sample(C, n_samples=n_samples-1)
    f = theano.function([H], samples, updates=updates_s)
    print f(h)