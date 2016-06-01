'''Extra operations for fMRI and MRI analysis.

'''

import numpy as np
import theano
import theano.tensor as T

from cortex.utils import floatX


class Detrender(theano.Op):
    '''Detrender for time courses.

    Detrends along the 0-axis. Wraps numpy.polyfit and numpy.polyval.

    Attributes:
        order (int): order of the polynomial fit.

    '''
    __props__ = ()

    itypes = [T.ftensor3]
    otypes = [T.ftensor3]

    def __init__(self, order=4):
        '''Initializer for Detrender.

        Args:
            order (int): order of the polynomial fit.

        '''
        self.order=order
        super(Detrender, self).__init__()

    def perform(self, node, inputs, output_storage):
        '''Perform detrending.

        '''
        data = inputs[0]
        z = output_storage[0]
        x = np.arange(data.shape[0])
        if len(data.shape) == 3:
            reshape = data.shape
            data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
        elif len(data.shape) > 3:
            raise ValueError('Detrending over 3 dims not supported')
        else:
            reshape = None
        fit = np.polyval(np.polyfit(x, data, deg=self.order),
                         np.repeat(x[:, None], data.shape[1], axis=1))
        data = data - fit
        if reshape is not None:
            data = data.reshape(reshape)
        z[0] = data.astype(floatX)

    def infer_shape(self, node, i0_shapes):
        return i0_shapes