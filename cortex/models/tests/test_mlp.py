'''Module for testing MLPs.

'''

from collections import OrderedDict
import numpy as np
from pprint import pformat, pprint
import theano
from theano import tensor as T

import cortex
from cortex import models
from cortex.datasets.basic.euclidean import Euclidean
from cortex.models import mlp as module
from cortex.utils import floatX, logger as cortex_logger


cortex_logger.set_stream_logger(2)
_atol = 1e-6
manager = cortex._manager

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

def feed_numpy_d(dmlp, x):
    mlp = dmlp.mlp
    dist = dmlp.distribution
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

    z = dist._act(z, as_numpy=True)
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
    data_iter = Euclidean()
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
    session = manager.create_session()
    session.build()

    f = theano.function(session.inputs, session.cost)
    data = session.next_batch(batch_size=10, mode='train')
    cost = f(*data)
    y = feed_numpy(manager.cells['mlp1'], data[0])
    y = feed_numpy(manager.cells['mlp2'], y[-1])

    _cost = ((y[-1] - data[0]) ** 2).mean()
    assert (abs(cost - _cost) <= _atol), abs(cost - _cost)
    logger.debug('Expected value of autoencoder cost OK within %.2e' % _atol)

def test_make_prob_autoencoder():
    manager.reset()
    data_iter = Euclidean()
    manager.prepare_cell('MLP', name='mlp1', dim_hs=[5, 7])
    manager.prepare_cell('DistributionMLP', name='mlp2', dim_in=13, dim_hs=[3, 11])
    manager.add_step('mlp1', 'fibrous.input')
    manager.add_step('mlp2', 'mlp1.output')
    manager.match_dims('mlp2.P', 'fibrous.input')
    manager.build()

def gaussian_nll(x, mu, log_sigma):
    return 0.5 * (
        (x - mu)**2 / (np.exp(2 * log_sigma)) +
        2 * log_sigma + np.log(2 * np.pi)).sum(axis=-1).mean()

def logistic_nll(x, mu, log_s):
    g = (x - mu) / np.exp(log_s)
    return (-g + log_s + 2 * np.log(1 + np.exp(g))).sum(axis=-1).mean()

def test_prob_autoencoder_graph():
    manager.reset()
    test_make_prob_autoencoder()
    manager.add_cost('mlp2.negative_log_likelihood', X='fibrous.input')
    session = manager.create_session()
    session.build()

    f = theano.function(session.inputs, session.cost)
    data = session.next_batch(batch_size=10, mode='train')
    cost = f(*data)
    y = feed_numpy(manager.cells['mlp1'], data[0])
    y = feed_numpy(manager.cells['mlp2.mlp'], y[-1])[-1]
    mu = y[:, :2]
    log_sigma = y[:, 2:]
    _cost = gaussian_nll(data[0], mu, log_sigma)
    assert (abs(cost - _cost) <= _atol), abs(cost - _cost)
    logger.debug('Expected value of probabilistic autoencoder cost OK within %.2e' % _atol)

def test_vae(prior='gaussian'):
    manager.reset()
    manager.prepare_data('dummy', name='data', n_samples=103, data_shape=(13,))
    manager.prepare_cell('DistributionMLP', name='approx_posterior',
                         dim_hs=[27], h_act='softplus')
    manager.prepare_cell(prior, name='prior', dim=5)
    manager.prepare_cell('DistributionMLP', name='conditional',
                         dim_hs=[23], h_act='softplus')
    manager.match_dims('prior.P', 'approx_posterior.P')
    manager.match_dims('conditional.P', 'data.input')
    manager.add_step('approx_posterior', 'data.input')
    manager.add_step('conditional', 'approx_posterior.samples')
    manager.prepare_samples('approx_posterior.P', 5)
    manager.build()

    manager.add_cost('conditional.negative_log_likelihood', X='data.input')
    manager.add_cost('kl_divergence', P='approx_posterior.P', Q='prior',
                     P_samples='approx_posterior.samples',
                     cells=['approx_posterior.distribution', 'prior'])

    session = manager.create_session()
    session.build(test=True)
    f = theano.function(session.inputs, [
        session.tensors['conditional.P'], session.tensors['approx_posterior.samples'], session.cost] + session.costs.values())
    data = session.next_batch(batch_size=11)
    p, samples, cost, kl_term, nll_term = f(*data)

    q = feed_numpy_d(manager.cells['approx_posterior'], data[0])[-1]
    py_h = feed_numpy_d(manager.cells['conditional'], samples)[-1]

    _nll_term = (-data[0][None, :, :] * np.log(py_h) -
                 (1 - data[0]) * np.log(1. - py_h)).sum(axis=-1).mean()

    if prior == 'gaussian':
        mu_q = q[:, :5]
        log_sigma_q = q[:, 5:]
        mu_pr = manager.cells['prior'].mu.get_value()
        log_sigma_pr = manager.cells['prior'].log_sigma.get_value()
        _kl_term = (log_sigma_pr - log_sigma_q + 0.5 * (
            (np.exp(2 * log_sigma_q) + (mu_pr[None, :] - mu_q) ** 2) /
            np.exp(2 * log_sigma_pr[None, :])
            - 1)).sum(axis=-1).mean()
    elif prior == 'logistic':
        mu_q = q[:, :5]
        log_s_q = q[:, 5:]
        mu_pr = manager.cells['prior'].mu.get_value()
        log_s_pr = manager.cells['prior'].log_s.get_value()
        neg_term = logistic_nll(samples, mu_q[None, :, :], log_s_q[None, :, :])
        pos_term = logistic_nll(
            samples, mu_pr[None, None, :], log_s_pr[None, None, :])
        _kl_term = (pos_term - neg_term)

    assert (abs(_kl_term - kl_term) <= 1e-5), (
        _kl_term - kl_term, _kl_term, kl_term)
    assert (abs(_nll_term - nll_term) <= 1e-5), (
        _nll_term - nll_term, _nll_term, nll_term)

    _cost = _nll_term + _kl_term

    assert (abs(cost - _cost) <= 1e-5), (_cost - cost)
    logger.debug('Expected value of VAE with prior %s cost OK within %.2e. '
                 % (prior, 1e-5))

def test_vae_logistic():
    test_vae(prior='logistic')