'''
Module for testing MLPs.
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from cortex import layers
from cortex.layers import mlp as module
from cortex.utils import floatX


sigmoid = 'lambda x: 1.0 / (1.0 + np.exp(-x))'
tanh = 'lambda x: np.tanh(x)'
softplus = 'lambda x: np.log(1.0 + np.exp(x))'


def test_fetch_class():
    C = layers.resolve_class('MLP')
    return C

def test_make_mlp(dim_in=13, dim_h=17, dim_out=19, n_layers=2,
                  h_act='T.nnet.softplus', distribution='binomial'):
    C = test_fetch_class()
    mlp = C(dim_in, dim_h, dim_out, n_layers, h_act=h_act,
            distribution=distribution)
    mlp.set_tparams()
    return mlp

def test_mlp_factory(dim_in=13, dim_hs=[17, 23], dim_out=19,
                     h_act='T.nnet.softplus', distribution='binomial'):
    C = test_fetch_class()
    return C.factory(
        dim_in=dim_in, dim_hs=dim_hs, dim_out=dim_out, h_act=h_act,
        distribution=distribution)

def test_feed_forward(mlp=None, X=T.matrix('X', dtype=floatX), x=None, distribution='binomial'):
    if mlp is None:
        mlp = test_make_mlp()
    Z = mlp.preact(X)
    Y = mlp.feed(X)

    batch_size = 23
    if x is None:
        x = np.random.randint(0, 2, size=(batch_size, mlp.dim_in)).astype(floatX)

    z = x
    for l in xrange(mlp.n_layers):
        W = mlp.params['W%d' % l]
        b = mlp.params['b%d' % l]

        z = np.dot(z, W) + b
        if l != mlp.n_layers - 1:
            activ = mlp.h_act
            if activ == 'T.nnet.sigmoid':
                activ = sigmoid
            elif activ == 'T.tanh':
                activ = tanh
            elif activ == 'T.nnet.softplus':
                activ = softplus
            elif activ == 'lambda x: x':
                pass
            else:
                raise ValueError(activ)
            z = eval(activ)(z)
            assert not np.any(np.isnan(z))

    if distribution == 'binomial':
        activ = sigmoid
    elif distribution == 'centered_binomial':
        activ = sigmoid
    elif distribution == 'gaussian':
        activ = 'lambda x: x'
    else:
        raise ValueError(distribution)
    print distribution, activ
    print
    y = eval(activ)(z)
    assert not np.any(np.isnan(y))

    f = theano.function([X], Y)
    y_test = f(x)
    assert not np.any(np.isnan(y_test)), y_test

    assert y.shape == y_test.shape, (y.shape, y_test.shape)

    assert np.allclose(y, y_test, atol=1e-4), (np.max(np.abs(y - y_test)))

    return OrderedDict(y=y, preact=z, Y=Y, Preact=Z)
