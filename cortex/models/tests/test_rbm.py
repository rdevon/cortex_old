'''
Module for testing MLPs.
'''

from collections import OrderedDict
import numpy as np
from pprint import pformat, pprint
import theano
from theano import tensor as T

import cortex
from cortex import models
from cortex.datasets.basic.dummy import Dummy
from cortex.models import mlp as module
from cortex.utils import floatX, logger as cortex_logger


sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
tanh = lambda x: np.tanh(x)
softplus = lambda x: np.log(1.0 + np.exp(x))
identity = lambda x: x

def test_fetch_class(c='RBM'):
    C = cortex.resolve_class(c)
    return C

def test_make_rbm(dim_in=13, dim_h=17):
    C = test_fetch_class()
    mlp = C(dim_in, dim_h)
    return mlp

def test_mlp_factory(dim_in=13, dim_h=17):
    cortex.reset()
    C = test_fetch_class()
    return C.factory(dim_in=dim_in, dim_h=dim_h)

def test_feed(dim_h=17):
    cortex.reset()
    cortex.prepare_data('dummy', name='data', n_samples=103, data_shape=(5,))
    cortex.prepare_cell('RBM', name='rbm', dim_h=3, h_dist_type='binomial')
    cortex.match_dims('rbm.input', 'data.input')
    cortex.add_step('rbm', 'data.input', n_steps=1, n_chains=3)
    
    cortex.build()
    cortex.profile()
    train_session = cortex.create_session()
    cortex.build_session(test=True)
    tensors = OrderedDict((k, v) for k, v in train_session.tensors.items()
        if not 'outputs' in k)
    
    f = theano.function(train_session.inputs, tensors,
                        updates=train_session.updates)
    data = train_session.next_batch(batch_size=11)
    rbm = cortex._manager.cells['rbm']
    W = rbm.W.get_value()
    b = rbm.v_dist.z.get_value()
    c = rbm.h_dist.z.get_value()
    
    rvals = f(*data)
    print rvals['rbm.pH0']
    
    ph0 = sigmoid(np.dot(rvals['rbm.V0'], W) + c) * .9999 + 5e-6
    print ph0
    print rvals['rbm.pH0'] - ph0
    assert False, rvals.keys()