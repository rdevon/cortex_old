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


logger = logging.getLogger(__name__)
cortex_logger.set_stream_logger(2)
_atol = 1e-6
manager = cortex._manager

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
tanh = lambda x: np.tanh(x)
softplus = lambda x: np.log(1.0 + np.exp(x))
identity = lambda x: x

def test_fetch_class(c='CNN2D'):
    C = cortex.resolve_class(c)
    return C

def test_make_cnn(input_shape=(19, 19), filter_shapes=((5, 5), (3, 3)),
                  pool_sizes=((2, 2), (2, 2)), n_filters=[100, 200],
                  h_act='softplus'):
    C = test_fetch_class()
    cnn = C(input_shape=input_shape, filter_shapes=filter_shapes,
            pool_sizes=pool_sizes, n_filters=n_filters, h_act=h_act)
    return cnn