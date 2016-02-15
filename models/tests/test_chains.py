'''
Tests for chains (data assignment with RNN)
'''

from collections import OrderedDict
import numpy as np
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import tensor as T

from datasets.chains import Chains
from datasets.euclidean import Euclidean
from models.chainer import RNNChainer
from models.rnn import RNN
from utils.tools import (
    floatX
)


sigmoid = lambda x: (1. / (1 + np.exp(-x))) * 0.9999 + 0.000005
neg_log_prob = lambda x, p: (-x * np.log(p) - (1 - x) * np.log(1 - p)).sum(axis=len(x.shape)-1)

def test_build_rnn(dim_in=31, dim_h=11, dim_out=None):
    print 'Building RNN'
    rnn = RNN(dim_in, dim_h, dim_out=dim_out)
    rnn.set_tparams()
    print 'RNN formed correctly'

    return rnn

def test_dataset(n_samples=2000, batch_size=13, dims=2):
    print 'Forming data iter'
    data_iter = Euclidean(n_samples=n_samples, dims=dims, batch_size=batch_size)
    print 'Data iter formed'
    return data_iter

def test_chain():
    X = T.tensor3('x', dtype=floatX)

    data_iter = test_dataset()
    rnn = test_build_rnn(dim_in=data_iter.dims[data_iter.name])
    chainer = RNNChainer(rnn)
    chain_dict, updates = chainer(X)

    f = theano.function([X], chain_dict.values(), updates=updates)
    x = data_iter.X[:, None, :]
    print f(x)

    chainer.build_data_chain(data_iter)

def test_chain_dataset():
    data_iter = test_dataset()
    chains = Chains(data_iter)

    rnn = test_build_rnn(dim_in=data_iter.dims[data_iter.name])
    chainer = RNNChainer(rnn)

    chains.set_chainer(chainer)
    print chains.next()

def test_assignment(dim_in=13, dim_h=17, n_samples=5, window=5):
    rnn = test_build_rnn(dim_in, dim_h)
    chainer = RNNChainer(rnn)

    data_iter = Euclidean(n_samples=n_samples, dims=dim_in, batch_size=window)

    x = data_iter.next()[data_iter.name]

    test_dict = OrderedDict()

    X = T.matrix('x', dtype=floatX)

    h0 = data_iter.rng.uniform(low=-1, high=-1, size=(rnn.dim_h,)).astype(floatX)
    H0 = theano.shared(h0)#T.alloc(0., rnn.dim_h).astype(floatX)
    P0 = rnn.output_net.feed(H0)
    p0 = sigmoid(np.dot(h0, rnn.output_net.params['W0']) + rnn.output_net.params['b0'])
    test_dict['output'] = (None, P0, None, p0, theano.OrderedUpdates())

    I0 = chainer.get_first_assign(X[:, None, :], P0)
    energy = neg_log_prob(x[:, None, :], p0)
    i0 = np.argmin(energy, axis=0)
    test_dict['first index'] = (X, I0, x, i0, theano.OrderedUpdates())

    C = T.zeros((X.shape[0], 1)).astype('int64')
    S = T.ones((X.shape[0], 1)).astype(floatX)
    C = T.set_subtensor(C[I0, T.arange(C.shape[1])], 1)
    S = T.set_subtensor(S[I0, T.arange(S.shape[1])], S[0] * 0)

    outs, _ = chainer.step_assign(
        I0, H0, C, S, X[:, None, :], 0, 1, *rnn.get_sample_params())

    I1, H1, P1, E1, C, S = outs

    c = np.zeros((x.shape[0], 1)).astype(floatX)
    s = np.ones_like(c)
    c[i0] = 1
    s[i0] = s[i0] = 0.

    x_p = x[:, None, :][i0, range(1)]
    y_p = np.dot(x_p, rnn.input_net.params['W0']) + rnn.input_net.params['b0']
    h1 = np.tanh(np.dot(h0, rnn.params['Ur']) + y_p)
    p1 = sigmoid(np.dot(h1, rnn.output_net.params['W0']) + rnn.output_net.params['b0'])
    x1 = p1

    es = neg_log_prob(x[:, None, :], p1)
    es = es - np.log(s)
    i1 = np.argmin(es, axis=0)
    c[i1] = 1
    s[i1] = 0
    e1 = es[i1]

    test_dict['next prob'] = (X, P1, x, p1, theano.OrderedUpdates())
    test_dict['next index'] = (X, I1, x, i1, theano.OrderedUpdates())
    test_dict['next energy'] = (X, E1, x, e1, theano.OrderedUpdates())

    for k, v in test_dict.iteritems():
        print 'Testing %s' % k
        inp, out, inp_np, out_np, updates = v
        if inp is None:
            inp = []
        else:
            inp = [inp]
        f = theano.function(inp, out, updates=updates)
        if inp_np is None:
            out_actual = f()
        else:
            out_actual = f(inp_np)
        if not np.allclose(out_np, out_actual):
            print 'np', out_np
            print 'theano', out_actual
            assert False, '%s failed' % k
        else:
            print 'np', out_np
            print 'theano', out_actual
            '%s tested OK' % k
