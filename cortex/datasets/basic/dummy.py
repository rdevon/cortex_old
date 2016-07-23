'''Dummy dataset for unit tests.

'''
import numpy as np

from .. import BasicDataset
from ...utils import floatX, intX


class Dummy(BasicDataset):
    def __init__(self, n_samples, data_shape, distribution='binomial',
                 name=None, **kwargs):
        if distribution == 'binomial':
            X = np.random.binomial(size=(n_samples,) + data_shape)
        elif distribution == 'gaussian':
            X = np.random.normal(size=(n_samples,) + data_shape)
        else:
            raise TypeError()
        if name is None:
            name = 'dummy_' + distribution

        data = {'input': X}
        distributions = {'input': distribution}
        super(Dummy, self).__init__(data, distributions, name=name,
                                    mode='train', **kwargs)

_classes = {'dummy', Dummy}