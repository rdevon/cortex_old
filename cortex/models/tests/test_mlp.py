'''
Module for testing MLPs.
'''

from collections import OrderedDict
import logging
import numpy as np
from pprint import pformat, pprint
import theano
from theano import tensor as T

import cortex
from cortex import models
from cortex.datasets.basic.euclidean import Euclidean
from cortex.models import mlp as module
from cortex.utils import floatX, logger as cortex_logger


logger = logging.getLogger(__name__)
cortex_logger.set_stream_logger(2)
_atol = 1e-6
manager = cortex.manager

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
tanh = lambda x: np.tanh(x)
softplus = lambda x: np.log(1.0 + np.exp(x))
identity = lambda x: x

def test_fetch_class(c='MLP'):
    C = cortex.resolve_class(c)
    return C

def test_make_mlp(dim_in=13, dim_h=17, dim_out=19, n_layers=2,
                  h_act='softplus', out_act='sigmoid'):
    C = test_fetch_class()
    mlp = C(dim_in, dim_out, dim_h=dim_h, n_layers=n_layers, h_act=h_act,
            out_act=out_act)
    return mlp

def test_make_distmlp(dim_in=13, dim_h=17, dim_out=19, n_layers=2,
                      h_act='softplus', distribution_type='logistic'):
    C = test_fetch_class('DistributionMLP')
    mlp = C(dim_in=dim_in, dim=dim_out, distribution_type=distribution_type,
            dim_hs=[dim_h], h_act=h_act)
    return mlp

def test_mlp_factory(dim_in=13, dim_hs=[17, 23], dim_out=19,
                     h_act='softplus', out_act='sigmoid'):
    C = test_fetch_class()
    return C.factory(
        dim_in=dim_in, dim_hs=dim_hs, dim_out=dim_out, h_act=h_act,
        out_act=out_act)

def convert_t_act(activ):
    if activ == T.nnet.sigmoid:
        activ = sigmoid
    elif activ == T.tanh:
        activ = tanh
    elif activ == T.nnet.softplus:
        activ = softplus
    elif activ(1234) == 1234:
        pass
    else:
        raise ValueError(activ)
    return activ

def feed_numpy(mlp, x):
    zs = []
    z = x
    for l in xrange(mlp.n_layers):
        W = mlp.params['weights'][l]
        b = mlp.params['biases'][l]

        z = np.dot(z, W) + b
        if l != mlp.n_layers - 1:
            activ = convert_t_act(mlp.h_act)
            z = activ(z)
            assert not np.any(np.isnan(z))
        elif mlp.out_act is not None:
            activ = convert_t_act(mlp.out_act)
            z = activ(z)
        zs.append(z)

    return zs

def _test_feed_forward(mlp=None, X=T.matrix('X', dtype=floatX), x=None,
                       out_act='sigmoid', batch_size=23):
    logger.debug('Testing feed forward with out act %s' % out_act)
    if mlp is None:
        mlp = test_make_mlp(out_act=out_act)
    if x is None:
        x = np.random.randint(0, 2, size=(batch_size, mlp.dim_in)).astype(floatX)

    outs = mlp(X)
    zs = feed_numpy(mlp, x)
    y = zs[-1]

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

def test_feed_forward_dmlp(mlp=None, X=T.matrix('X', dtype=floatX), x=None,
                           distribution_type='binomial'):
    manager.reset()
    if mlp is None:
        mlp = test_make_distmlp(distribution_type=distribution_type)
    outs = mlp(X)
    batch_size = 23
    if x is None:
        x = np.random.randint(0, 2, size=(batch_size, mlp.dim_in)).astype(floatX)

    z = x
    for l in xrange(mlp.n_layers):
        W = mlp.mlp.params['weights'][l]
        b = mlp.mlp.params['biases'][l]

        z = np.dot(z, W) + b
        if l == mlp.n_layers - 1:
            Z = outs['Y']
        else:
            activ = convert_t_act(mlp.h_act)
            z = activ(z)
            assert not np.any(np.isnan(z))
            Z = outs['H_%d' % l]

        f = theano.function([X], Z)
        z_ = f(x)
        assert np.allclose(z, z_, atol=_atol)

    if distribution_type == 'binomial':
        activ = lambda x: sigmoid(x) * 0.9999 + 0.000005

    y = activ(z)
    assert not np.any(np.isnan(y))
    logger.debug('No nan values found in numpy test')

    f = theano.function([X], outs['P'])
    y_test = f(x)
    assert not np.any(np.isnan(y_test)), y_test
    logger.debug('No nan values found in theano test')

    assert y.shape == y_test.shape, (y.shape, y_test.shape)
    logger.debug('Shapes match.')

    assert np.allclose(y, y_test, atol=_atol)
    logger.debug('Expected value of MLP feed forward OK within %.2e'
                        % _atol)

def test_make_autoencoder(dim_in=13):
    manager.reset()
    data_iter = Euclidean(batch_size=10)
    manager.prepare_cell('MLP', name='mlp1', dim_hs=[5, 7])
    manager.prepare_cell('MLP', name='mlp2', dim_in=dim_in, dim_hs=[3, 11])
    manager.add_step('mlp1', 'fibrous.input')
    manager.add_step('mlp2', 'mlp1.output')
    manager.match_dims('mlp2.output', 'fibrous.input')
    manager.build()

def test_autoencoder_graph():
    manager.reset()
    test_make_autoencoder()
    manager.add_cost('squared_error', Y_hat='mlp2.output', Y='fibrous.input')
    session = manager.build_session()

    f = theano.function(session.inputs, sum(session.costs))
    data = session.next(mode='train')
    cost = f(*data)
    y = feed_numpy(manager.cells['mlp1'], data[0])
    y = feed_numpy(manager.cells['mlp2'], y[-1])
    _cost = ((y[-1] - data[0]) ** 2).mean()
    assert (abs(cost - _cost) <= _atol), abs(cost - _cost)
    logger.debug('Expected value of autoencoder cost OK within %.2e' % _atol)

def test_make_prob_autoencoder():
    manager.reset()
    data_iter = Euclidean(batch_size=10)
    manager.prepare_cell('MLP', name='mlp1', dim_hs=[5, 7])
    manager.prepare_cell('DistributionMLP', name='mlp2', dim_in=13, dim_hs=[3, 11])
    manager.add_step('mlp1', 'fibrous.input')
    manager.add_step('mlp2', 'mlp1.output')
    manager.match_dims('mlp2.P', 'fibrous.input')
    manager.build()

def test_prob_autoencoder_graph():
    manager.reset()
    test_make_prob_autoencoder()
    manager.add_cost('mlp2.negative_log_likelihood', X='fibrous.input')
    session = manager.build_session()

    f = theano.function(session.inputs, sum(session.costs))
    data = session.next(mode='train')
    cost = f(*data)
    y = feed_numpy(manager.cells['mlp1'], data[0])
    y = feed_numpy(manager.cells['mlp2.mlp'], y[-1])[-1]
    mu = y[:, :2]
    log_sigma = y[:, 2:]
    _cost = 0.5 * (
        (data[0] - mu)**2 / (np.exp(2 * log_sigma)) +
        2 * log_sigma + np.log(2 * np.pi)).sum(axis=1).mean()
    assert (abs(cost - _cost) <= _atol), abs(cost - _cost)
    logger.debug('Expected value of probabilistic autoencoder cost OK within %.2e' % _atol)

def test_vae():
    manager.reset()
    manager.prepare_data('dummy', name='data', batch_size=11, n_samples=103,
                         data_shape=(13,))
    manager.prepare_cell('DistributionMLP', name='approx_posterior',
                         dim_hs=[27], h_act='softplus')
    manager.prepare_cell('gaussian', name='prior', dim=5)
    manager.prepare_cell('DistributionMLP', name='conditional',
                         dim_hs=[23], h_act='softplus')
    manager.match_dims('prior.P', 'approx_posterior.P')
    manager.match_dims('conditional.P', 'data.input')
    manager.add_step('approx_posterior', 'data.input')
    manager.add_step('conditional', 'approx_posterior.samples')
    manager.build()

    manager.generate_samples('conditional', n_samples=7)
    manager.add_cost('conditional.negative_log_likelihood', X='data.input')
    manager.add_cost('prior.kl_divergence', Q='approx_posterior.P')