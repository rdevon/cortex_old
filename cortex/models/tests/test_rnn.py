'''
Tests for RNN
'''

from collections import OrderedDict
import logging
import numpy as np
import theano
from theano import tensor as T

import cortex
from cortex.datasets.basic.dummy import Dummy
from cortex.models import rnn as rnn_module
from cortex.models.tests.test_mlp import feed_numpy as feed_numpy_mlp, feed_numpy_d as feed_numpy_dmlp
from cortex.utils import floatX, logger as cortex_logger


logger = logging.getLogger(__name__)
#cortex_logger.set_stream_logger(2)
_atol = 1e-6
manager = cortex.manager

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
tanh = lambda x: np.tanh(x)
softplus = lambda x: np.log(1.0 + np.exp(x))
identity = lambda x: x

def _test_fetch_class(c='RecurrentUnit'):
    C = cortex.resolve_class(c)
    return C

def test_fetch_classes():
    for c in ['RecurrentUnit', 'RNNInitializer', 'RNN', 'GenRNN']:
        _test_fetch_class(c=c)

def test_make_rnn(dim_in=13, dim_h=17, initialization='MLP'):
    C = _test_fetch_class('RNN')
    rnn = C(dim_in=dim_in, dim_h=dim_h, initialization=initialization)
    return rnn

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

def feed_numpy(rnn, x, m=None):
    if m is None: m = np.ones((x.shape[0], x.shape[1]))
    y = feed_numpy_mlp(rnn.input_net, x)[-1]
    h_ = feed_numpy_mlp(rnn.initializer.initializer, x[0])[-1]
    W = rnn.RU.params['W']
    hs = []
    for i in range(x.shape[0]):
        preact = np.dot(h_, W) + y[i]
        h = tanh(preact)
        h_ = m[i][:, None] * h + (1 - m[i])[:, None] * h_
        hs.append(h_)

    return np.array(hs)

def feed_numpy_genrnn(rnn, x, m=None):
    if m is None: m = np.ones((x.shape[0], x.shape[1]))
    y = feed_numpy_mlp(rnn.input_net, x)[-1]
    h_ = feed_numpy_mlp(rnn.initializer.initializer, x[0])[-1]
    W = rnn.RU.params['W']
    hs = []
    for i in range(x.shape[0]):
        preact = np.dot(h_, W) + y[i]
        h = tanh(preact)
        h_ = m[i][:, None] * h + (1 - m[i])[:, None] * h_
        hs.append(h_)

    hs = np.array(hs)
    ps = feed_numpy_dmlp(rnn.output_net, hs)[-1]
    return ps

def test_feed(rnn=None, X=T.tensor3('X', dtype=floatX), x=None, batch_size=23,
              n_steps=7):
    manager.reset()
    logger.debug('Testing feed forward')
    if rnn is None:
        rnn = test_make_rnn()
    if x is None:
        x = np.random.randint(
            0, 2, size=(n_steps, batch_size, rnn.dim_in)).astype(floatX)

    h = feed_numpy(rnn, x)
    outs = rnn(X)
    f = theano.function([X], outs['H'], updates=outs['updates'])
    h_test = f(x)
    assert not np.any(np.isnan(h_test)), h_test
    logger.debug('No nan values found in theano test')

    assert h.shape == h_test.shape, (h.shape, h_test.shape)
    logger.debug('Shapes match.')

    assert np.allclose(h, h_test, atol=_atol), (np.max(np.abs(h - h_test)))
    logger.debug('Expected value of RNN feed OK within %.2e'
                 % _atol)

def test_make_rnn_graph():
    manager.reset()
    manager.prepare_data('dummy', batch_size=11, n_samples=103, data_shape=(37, 2),
                         transpose={'input': (1, 0, 2)})
    manager.prepare_data('dummy', batch_size=11, n_samples=103, data_shape=(2,),
                         distribution='gaussian')
    manager.prepare_cell('RNN', name='rnn', dim_h=17, initialization='MLP')
    manager.prepare_cell('DistributionMLP', name='mlp')
    manager.add_step('rnn', 'dummy_binomial.input')
    manager.add_step('mlp', 'rnn.output')
    manager.match_dims('mlp.P', 'dummy_gaussian.input')
    manager.build()

def test_rnn_cost():
    test_make_rnn_graph()
    manager.add_cost('mlp.negative_log_likelihood', X='dummy_gaussian.input')
    session = manager.build_session(test=True)

    f = theano.function(session.inputs, sum(session.costs), updates=session.updates)
    data = session.next(mode='train')
    cost = f(*data)
    hs = feed_numpy(manager.cells['rnn'], data[0])
    y = feed_numpy_mlp(manager.cells['mlp'].mlp, hs[-1])[-1]
    mu = y[:, :2]
    log_sigma = y[:, 2:]
    _cost = 0.5 * (
        (data[1] - mu)**2 / (np.exp(2 * log_sigma)) +
        2 * log_sigma + np.log(2 * np.pi)).sum(axis=1).mean()
    assert (abs(cost - _cost) <= _atol), abs(cost - _cost)
    logger.debug('Expected value of RNN cost OK within %.2e' % _atol)

def test_make_genrnn_graph():
    manager.reset()
    manager.prepare_data('dummy', batch_size=3, n_samples=103, data_shape=(5, 2),
                         transpose={'input': (1, 0, 2)})
    manager.prepare_cell('GenRNN', name='rnn', dim_h=17, initialization='MLP')
    manager.add_step('rnn', 'dummy_binomial.input')
    manager.match_dims('rnn.P', 'dummy_binomial.input')
    manager.build()

def test_genrnn_cost():
    test_make_genrnn_graph()
    manager.add_cost('rnn.negative_log_likelihood',
                     X='dummy_binomial.input')
    session = manager.build_session(test=True)
    f = theano.function(session.inputs, sum(session.costs), updates=session.updates)
    data = session.next(mode='train')
    cost = f(*data)
    f = theano.function(session.inputs, session.tensors['rnn.P'], updates=session.updates)
    p_ = f(*data)

    p = feed_numpy_genrnn(manager.cells['rnn'], data[0])
    print p - p_
    x = data[0][1:]
    p = p[:-1]
    p = np.clip(p, 1e-7, 1-1e-7)
    _cost = -(x * np.log(p) + (1. - x) * np.log(1. - p)).sum(axis=-1).sum(axis=0).mean()
    assert (abs(cost - _cost) <= _atol), (cost, _cost, abs(cost - _cost))
    logger.debug('Expected value of genrnn cost OK within %.2e' % _atol)

def _test_sample(dim_in=13, dim_h=17, n_samples=107, n_steps=21):
    rnn = test_build(dim_in, dim_h)
    results, updates = rnn.sample(n_samples=107, n_steps=21)
    f = theano.function([], results['p'], updates=updates)
    f()

def _test_recurrent(dim_in=13, dim_h=17, n_samples=107, window=7):
    rnn = test_build(dim_in, dim_h)

    data_iter = Euclidean(n_samples=n_samples, dims=dim_in, batch_size=window)
    x = data_iter.next()[data_iter.name]

    test_dict = OrderedDict()

    X = T.matrix('x', dtype=floatX)

    Y = rnn.call_seqs(X, None, 0, *rnn.get_sample_params())[0]
    y = np.dot(x, rnn.input_net.params['W0']) + rnn.input_net.params['b0']
    test_dict['RNN preact from data'] = (X, Y, x, y, theano.OrderedUpdates())
    H0 = T.alloc(0., X.shape[0], rnn.dim_hs[0]).astype(floatX)
    H = rnn._step(1, Y, H0, rnn.Ur0)
    h0 = np.zeros((x.shape[0], rnn.dim_hs[0])).astype(floatX)
    h = np.tanh(np.dot(h0, rnn.params['Ur0']) + y)
    test_dict['step reccurent'] = (X, H, x, h, theano.OrderedUpdates())

    P = rnn.output_net.feed(H)
    p = sigmoid(np.dot(h, rnn.output_net.params['W0']) + rnn.output_net.params['b0'])
    test_dict['output'] = (X, P, x, p, theano.OrderedUpdates())

    for k, v in test_dict.iteritems():
        print 'Testing %s' % k
        inp, out, inp_np, out_np, updates = v
        f = theano.function([inp], out, updates=updates)
        out_actual = f(inp_np)
        if not np.allclose(out_np, out_actual):
            print 'np', out_np
            print 'theano', out_actual
            assert False
