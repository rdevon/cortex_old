'''Dummy dataset for unit tests.

'''
import numpy as np

from .. import BasicDataset
from ...utils import floatX, intX


class Dummy(BasicDataset):
    def __init__(self, n_samples, data_shape, distribution='binomial',
                 name=None, mode='train', **kwargs):
        shape = list((n_samples,) + data_shape)
        if distribution == 'binomial':
            X = np.random.binomial(
                p=0.5, size=shape, n=1).astype(floatX)
        elif distribution == 'gaussian':
            X = np.random.normal(size=shape).astype(floatX)
        else:
            raise TypeError()
        if name is None:
            name = 'dummy_' + distribution

        data = {'input': X}
        distributions = {'input': distribution}
        super(Dummy, self).__init__(data, distributions, name=name,
                                    mode=mode, **kwargs)

_classes = {'dummy': Dummy}