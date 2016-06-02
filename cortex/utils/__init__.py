'''Init for utils.

'''

import numpy as np
import theano
import warnings


warnings.filterwarnings("ignore", module="nipy")
intX = 'int64'
floatX = theano.config.floatX
pi = theano.shared(np.pi).astype(floatX)
e = theano.shared(np.e).astype(floatX)