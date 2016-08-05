'''Init for utils.

'''

import numpy as np
import random
import theano
import warnings

from .base import *

__all__ = ['base']

_random_seed = random.randint(1, 10000)
_rng = np.random.RandomState(_random_seed)
intX = 'int64'
floatX = theano.config.floatX
pi = theano.shared(np.pi).astype(floatX)
e = theano.shared(np.e).astype(floatX)

warnings.filterwarnings('ignore', module='nipy')
warnings.filterwarnings('ignore',
                        'This call to matplotlib.use() has no effect.*',
                        UserWarning)