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
from utils.tools import floatX


sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

def test_make_gru(dim_in=31, dim_h=11, dim_out=None,
                  i_net=None, a_net=None, o_net=None, c_net=None):
    print 'Testing GRU formation'

    if i_net is None:
        i_net = dict(
            dim_h=17,
            n_layers=2,
            h_act='T.tanh',
            out_act='T.tanh',
            weight_scale=0.1,
        )
    if a_net is None:
        a_net = dict(
            dim_h=19,
            n_layers=2,
            h_act='T.tanh',
            out_act='T.tanh',
            weight_scale=0.1
        )
    if o_net is None:
        o_net = dict(
            dim_h=23,
            n_layers=2,
            weight_scale=0.1
        )

    nets = dict(i_net=i_net, a_net=a_net, o_net=o_net, c_net=c_net)

    trng = RandomStreams(101)

    mlps = GRU.mlp_factory(dim_in, dim_h, dim_out=dim_out, **nets)
    rnn = GRU(dim_in, dim_h, dim_out=dim_out, trng=trng, **mlps)
    rnn.set_tparams()
    print 'GRU formed correctly'

    return rnn

def test_step(rnn=None, X=T.tensor3('X', dtype=floatX), C=None,
              H0=T.matrix('H0', dtype=floatX), x=None, c=None, h0=None,
              window=5, batch_size=9):

    if rnn is None:
        rnn = test_make_gru()
    dim_in = rnn.dim_in
    dim_h = rnn.dim_h

    if x is None:
        x = np.random.randint(0, 2, size=(window, batch_size, dim_in)).astype(floatX)
    if h0 is None:
        h0 = np.random.normal(loc=0, scale=1.0, size=(x.shape[1], dim_h)).astype(floatX)
    if c is not None and C is None:
        C = T.matrix('C', dtype=floatX)

    input_dict = test_mlp.test_feed_forward(mlp=rnn.input_net, X=X, x=x)
    aux_dict = test_mlp.test_feed_forward(mlp=rnn.input_net_aux, X=X, x=x)

    H1 = rnn._step(aux_dict['Preact'][0], input_dict['Preact'][0], H0,
                   *rnn.get_params())

    def step(h_, y_a, y_i):
        preact = np.dot(h_, rnn.params['Ura']) + y_a
        r = sigmoid(preact[:, :rnn.dim_h])
        u = sigmoid(preact[:, rnn.dim_h:])
        preactx = np.dot(h_, rnn.params['Urb']) * r + y_i
        h = np.tanh(preactx)
        h = u * h + (1. - u) * h_
        return h

    h = step(h0, aux_dict['preact'][0], input_dict['preact'][0])

    f = theano.function([X, H0], H1)
    h_test = f(x, h0)
    assert np.allclose(h_test, h, atol=1e-7), (np.max(np.abs(h_test - h)))

    rnn_dict, updates = rnn(X, h0=H0, condition_on=C)
    tinps = [X, H0]
    inps = [x, h0]
    if C is not None:
        tinps.append(C)
        inps.append(c)
    f = theano.function(tinps, rnn_dict.values(), updates=updates)

    vals = f(*inps)
    v_dict = OrderedDict((k, v) for k, v in zip(rnn_dict.keys(), vals))

    hs = []
    h = h0
    for t in xrange(window):
        h = step(h, aux_dict['preact'][t], input_dict['preact'][t])
        hs.append(h)

    hs = np.array(hs).astype(floatX)

    assert np.allclose(hs, v_dict['h'], atol=1e-7), (hs, v_dict['h'])
    print 'RNN Hiddens test out'

    out_dict = test_mlp.test_feed_forward(
        mlp=rnn.output_net, X=T.tensor3('H', dtype=floatX), x=hs)

    if c is not None:
        cond_dict = test_mlp.test_feed_forward(
            mlp=rnn.conditional, X=C, x=c)
        z = cond_dict['preact'] + out_dict['preact']
        activ = rnn.output_net.out_act
        if activ == 'T.nnet.sigmoid':
            activ = 'sigmoid'
        elif activ == 'T.tanh':
            activ = 'tanh'
        elif activ == 'lambda x: x':
            pass
        else:
            raise ValueError(activ)
        p = eval(activ)(z)
    else:
        p = out_dict['y']

    assert np.allclose(p, v_dict['p'], atol=1e-7), (p - v_dict['p'])

    return OrderedDict(p=p, P=v_dict['p'])
