'''
Module for testing GRU.
'''

from collections import OrderedDict
import numpy as np
import random
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import tensor as T

from models.gru import GRU
import test_mlp
from utils import floatX


sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

def test_make_gru(dim_in=31, dim_h=11, dim_out=None,
                  i_net=None, a_net=None, o_net=None, c_net=None):
    print 'Testing GRU formation'

    if i_net is None:
        i_net = dict(
            dim_h=17,
            n_layers=2,
            h_act='T.tanh',
            weight_scale=0.1,
        )
    if a_net is None:
        a_net = dict(
            dim_h=19,
            n_layers=2,
            h_act='T.tanh',
            weight_scale=0.1
        )
    if o_net is None:
        o_net = dict(
            dim_h=23,
            n_layers=2,
            weight_scale=0.1,
            distribution='binomial'
        )

    nets = dict(i_net=i_net, a_net=a_net, o_net=o_net, c_net=c_net)

    trng = RandomStreams(101)

    rnn = GRU.factory(dim_in=dim_in, dim_hs=[dim_h], dim_out=dim_out, **nets)
    rnn.set_tparams()
    print 'GRU formed correctly'

    return rnn

def test_step(rnn=None, X=T.tensor3('X', dtype=floatX),
              H0=T.matrix('H0', dtype=floatX), x=None, h0=None,
              window=5, batch_size=9):

    if rnn is None:
        rnn = test_make_gru()
    dim_in = rnn.dim_in
    dim_h = rnn.dim_hs[0]

    if x is None:
        x = np.random.randint(0, 2, size=(window, batch_size, dim_in)).astype(floatX)
    if h0 is None:
        h0 = np.random.normal(loc=0, scale=1.0, size=(x.shape[1], dim_h)).astype(floatX)

    input_dict = test_mlp.test_feed_forward(mlp=rnn.input_net, X=X, x=x,
                                            distribution='centered_binomial')
    aux_dict = test_mlp.test_feed_forward(mlp=rnn.input_net_aux, X=X, x=x,
                                          distribution='centered_binomial')

    H1 = rnn._step(1, aux_dict['Preact'][0], input_dict['Preact'][0], H0,
                   *rnn.get_params())

    def step(h_, y_a, y_i):
        preact = np.dot(h_, rnn.params['Ura0']) + y_a
        r = sigmoid(preact[:, :rnn.dim_hs[0]])
        u = sigmoid(preact[:, rnn.dim_hs[0]:])
        preactx = np.dot(h_, rnn.params['Urb0']) * r + y_i
        h = np.tanh(preactx)
        h = u * h + (1. - u) * h_
        return h

    h = step(h0, aux_dict['preact'][0], input_dict['preact'][0])

    f = theano.function([X, H0], H1)
    h_test = f(x, h0)
    assert np.allclose(h_test, h, atol=1e-7), (np.max(np.abs(h_test - h)))

    rnn_dict, updates = rnn(X, h0s=[H0])
    tinps = [X, H0]
    inps = [x, h0]
    rnn_values = [v[0] if isinstance(v, list) else v for v in rnn_dict.values()]

    f = theano.function(tinps, rnn_values, updates=updates)

    vals = f(*inps)
    v_dict = OrderedDict((k, v) for k, v in zip(rnn_dict.keys(), vals))

    hs = []
    h = h0
    for t in xrange(window):
        h = step(h, aux_dict['preact'][t], input_dict['preact'][t])
        hs.append(h)

    hs = np.array(hs).astype(floatX)

    assert np.allclose(hs[0], v_dict['hs'][0], atol=1e-7), (hs[0] - v_dict['hs'][0])
    print 'RNN Hiddens test out'

    out_dict = test_mlp.test_feed_forward(
        mlp=rnn.output_net, X=T.tensor3('H', dtype=floatX), x=hs)

    p = out_dict['y']

    assert np.allclose(p, v_dict['p'], atol=1e-4), (p - v_dict['p'])

    return OrderedDict(p=p, P=v_dict['p'])
