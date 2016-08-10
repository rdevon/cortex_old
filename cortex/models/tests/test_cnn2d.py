'''Tests for CNN

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
from cortex.models.tests.test_mlp import convert_t_act


_atol = 1e-6
manager = cortex._manager

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
tanh = lambda x: np.tanh(x)
softplus = lambda x: np.log(1.0 + np.exp(x))
identity = lambda x: x

def test_fetch_class(c='CNN2D'):
    C = cortex.resolve_class(c)
    return C

def test_make_cnn(input_shape=(3, 19, 19), filter_shapes=((5, 5), (3, 3)),
                  pool_sizes=((2, 2), (2, 2)), n_filters=[5, 7],
                  h_act='softplus'):
    C = test_fetch_class()
    cnn = C(input_shape=input_shape, filter_shapes=filter_shapes,
            pool_sizes=pool_sizes, n_filters=n_filters, h_act=h_act)
    return cnn

def test_make_cnn_classifier():
    cortex.reset()
    cortex.prepare_data('dummy', name='data', n_samples=103, data_shape=(3 * 31 * 31,))
    cortex.prepare_data('dummy', name='labels', n_samples=103, data_shape=(2,))
    cortex.prepare_cell('CNN2D', name='conv', input_shape=(3, 31, 31),
                        filter_shapes=((5, 5), (3, 3)),
                        pool_sizes=((2, 2), (2, 2)), n_filters=[5, 7],
                        h_act='softplus', out_act='identity')
    cortex.prepare_cell('DistributionMLP', name='ffn', dim_hs=[23])

    cortex.add_step('conv', 'data.input')
    cortex.add_step('ffn', 'conv.output')
    cortex.match_dims('ffn.P', 'labels.input')

    cortex.build()

def test_classifier():
    cortex.reset()
    test_make_cnn_classifier()

    cortex.add_cost('ffn.negative_log_likelihood', X='labels.input')

    session = cortex.create_session(batch_size=23)
    session.build(test=True)

def test_make_cnn_classifier2():
    cortex.reset()

    cortex.prepare_data('dummy', name='data', n_samples=103, data_shape=(3 * 31 * 31,))
    cortex.prepare_data('dummy', name='labels', n_samples=103, data_shape=(2,))
    cnn_args = dict(
        cell_type='CNN2D',
        input_shape=(3, 31, 31),
        filter_shapes=((5, 5), (3, 3), (2, 2)),
        pool_sizes=((2, 2), (5, 5), (1, 1)), n_filters=[5, 7, 2],
        h_act='softplus', out_act='identity'
    )
    cortex.prepare_cell('DistributionMLP', name='classifier',
                        mlp=cnn_args)

    cortex.add_step('classifier', 'data.input')
    cortex.match_dims('classifier.P', 'labels.input')

    cortex.build()

def test_classifier2():
    cortex.reset()
    test_make_cnn_classifier2()

    cortex.add_cost('classifier.negative_log_likelihood', X='labels.input')

    session = cortex.create_session(batch_size=23)
    session.build(test=True)