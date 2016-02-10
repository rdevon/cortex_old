'''
Tests for chains (data assignment with RNN)
'''

import numpy as np
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import tensor as T

from datasets.euclidean import Euclidean
from models.chainer import RNNChainer
from models.rnn import RNN
from utils.tools import (
    floatX
)


def test_build_rnn(dim_in=31, dim_h=11, dim_out=None, i_net=None, o_net=None):
    print 'Building RNN'

    if i_net is None:
        i_net = dict(
            dim_h=17,
            n_layers=2,
            h_act='T.tanh',
            distribution='centered_binomial',
            weight_scale=0.1,
        )
    if o_net is None:
        o_net = dict(
            dim_h=23,
            n_layers=2,
            h_act='T.tanh',
            weight_scale=0.1,
            distribution='continuous_binomial'
        )

    nets = dict(i_net=i_net, o_net=o_net)

    trng = RandomStreams(101)
    mlps = RNN.mlp_factory(dim_in, dim_h, dim_out=dim_out, **nets)
    rnn = RNN(dim_in, dim_h, dim_out=dim_out, trng=trng, **mlps)
    rnn.set_tparams()
    print 'RNN formed correctly'

    return rnn

def test_dataset(n_samples=2000, batch_size=13, dims=2):
    print 'Forming data iter'
    data_iter = Euclidean(n_samples=n_samples, dims=dims, batch_size=batch_size)
    print 'Data iterm formed'
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


