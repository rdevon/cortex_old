'''
Module for testing MLPs.
'''

from collections import OrderedDict
import logging
import numpy as np
import theano
from theano import tensor as T

import cortex
from cortex import models
from cortex.models import mlp as module
from cortex.utils import floatX
from cortex.utils import logger as cortex_logger


logger = logging.getLogger(__name__)
_atol = 1e-7

def test_fetch_class(c='MLP'):
    C = models.resolve_class(c)
    return C

def test_make_mlp(dim_in=13, dim_h=17, dim_out=19, n_layers=2,
                  h_act='softplus', out_act='sigmoid'):
    C = test_fetch_class()
    mlp = C(dim_in, dim_out, dim_h=dim_h, n_layers=n_layers, h_act=h_act,
            out_act=out_act)
    mlp.set_tparams()
    return mlp

def test_make_distmlp(dim_in=13, dim_h=17, dim_out=19, n_layers=2,
                      h_act='softplus', distribution_type='logistic'):
    C = test_fetch_class('DistributionMLP')
    mlp = C(dim_in, dim=dim_out, distribution_type=distribution_type,
            dim_hs=[dim_h], h_act=h_act)
    mlp.set_tparams()
    return mlp

def test_mlp_factory(dim_in=13, dim_hs=[17, 23], dim_out=19,
                     h_act='softplus', out_act='sigmoid'):
    C = test_fetch_class()
    return C.factory(
        dim_in=dim_in, dim_hs=dim_hs, dim_out=dim_out, h_act=h_act,
        out_act=out_act)

def _test_feed_forward(mlp=None, X=T.matrix('X', dtype=floatX), x=None,
                       out_act='sigmoid'):
    if mlp is None:
        mlp = test_make_mlp(out_act=out_act)
    outs = mlp(X)

    batch_size = 23
    if x is None:
        x = np.random.randint(0, 2, size=(batch_size, mlp.dim_in)).astype(floatX)

    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    tanh = lambda x: np.tanh(x)
    softplus = lambda x: np.log(1.0 + np.exp(x))
    identity = lambda x: x

    z = x
    for l in xrange(mlp.n_layers):
        W = mlp.params['weights'][l]
        b = mlp.params['biases'][l]

        z = np.dot(z, W) + b
        if l != mlp.n_layers - 1:
            activ = mlp.h_act
            if activ == T.nnet.sigmoid:
                activ = sigmoid
            elif activ == T.tanh:
                activ = tanh
            elif activ == T.nnet.softplus:
                activ = softplus
            elif activ == (lambda x: x):
                pass
            else:
                raise ValueError(activ)
            z = activ(z)
            assert not np.any(np.isnan(z))

    activ = eval(out_act)
    y = activ(z)
    assert not np.any(np.isnan(y))
    logger.debug('No nan values found in numpy test')

    f = theano.function([X], outs['Y'])
    y_test = f(x)
    assert not np.any(np.isnan(y_test)), y_test
    logger.debug('No nan values found in theano test')

    assert y.shape == y_test.shape, (y.shape, y_test.shape)
    logger.debug('Shapes match.')

    assert np.allclose(y, y_test, atol=_atol), (np.max(np.abs(y - y_test)))
    logger.debug('Expected value of MLP feed forward OK within %.2e'
                        % _atol)

def test_feed_forward():
    for out_act in ['sigmoid', 'softplus', 'tanh', 'identity']:
        _test_feed_forward(out_act=out_act)
