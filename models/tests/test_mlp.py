'''
Module for testing MLPs.
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from models.mlp import MLP
from utils.tools import floatX


sigmoid = 'lambda x: 1.0 / (1.0 + np.exp(-x))'
tanh = 'lambda x: np.tanh(x)'
softplus = 'lambda x: np.log(1.0 + np.exp(x))'


def test_make_mlp(dim_in=13, dim_h=17, dim_out=19, n_layers=2, h_act='T.nnet.softplus', out_act='T.nnet.sigmoid'):
    mlp = MLP(dim_in, dim_h, dim_out, n_layers, h_act=h_act, out_act=out_act)
    mlp.set_tparams()
    return mlp

def test_feed_forward(mlp=None, X=T.matrix('X', dtype=floatX), x=None):
    if mlp is None:
        mlp = test_make_mlp()
    Z = mlp(X, return_preact=True)
    Y = mlp(X)

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

    activ = mlp.out_act
    if activ == 'T.nnet.sigmoid':
        activ = sigmoid
    elif activ == 'T.tanh':
        activ = tanh
    elif activ == 'lambda x: x':
        pass
    else:
        raise ValueError(activ)
    y = eval(activ)(z)
    assert not np.any(np.isnan(y))

    f = theano.function([X], Y)
    y_test = f(x)
    assert not np.any(np.isnan(y_test)), y_test

    assert y.shape == y_test.shape, (y.shape, y_test.shape)

    assert np.allclose(y, y_test, atol=1e-6), (np.max(np.abs(y - y_test)))

    return OrderedDict(y=y, preact=z, Y=Y, Preact=Z)