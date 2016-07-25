'''Module for testing distributions.

'''

from collections import OrderedDict
import logging
import numpy as np
import theano
from theano import tensor as T

import cortex
from cortex.models.distributions import _classes as classes

logger = logging.getLogger(__name__)
_atol = 1e-6
manager = cortex.manager

def _test_fetch_class(c='binomial'):
    C = cortex.resolve_class(c)
    return C

def test_fetch_classes():
    for c in classes:
        _test_fetch_class(c)

def test_sample(c='binomial'):
    manager.reset()
    manager.prepare_cell(c, name='dist', dim=3)
    manager.prepare_samples('dist', 5)
    manager.build()
    session = manager.build_session()
    f = theano.function([], session.tensors['dist.samples_epsilon'])
    rval = f()
    f = theano.function([], session.tensors['dist.samples'])
    rval = f()
    assert (f().shape == (5, 3)), c
    logger.info('Test sample %s passed' % c)

def test_samples():
    for c in ['binomial', 'gaussian', 'logistic', 'laplace', 'centered_binomial']:
        test_sample(c)
