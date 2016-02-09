'''
SimTB dataset class.
'''

from collections import OrderedDict
import numpy as np

from . import Dataset
from utils.tools import (
    floatX,
    warn_kwargs
)


class SimTB(Dataset):
    def __init__(self, source=None, **kwargs):
        kwargs = super(SimTB, self).__init__(**kwargs)

        if source is None:
            raise ValueError('No source provided')

        # Fetch simTB data from "source" source can be file, directory, etc.
        self.X = self.get_data(source)
        self.n = self.X.shape[0]

        # Reference for the dimension of the dataset. A dict is used for
        # multimodal data (e.g., mri and labels)
        self.dims = dict()
        self.dims[self.name] = self.X.shape[1]

        # This is reference for models to decide how the data should be modelled
        # E.g. with a binomial or gaussian variable
        self.distributions = dict()
        self.distributions[self.name] = 'gaussian'

        # We will probably center the data in the main script using this
        # global mean image.
        self.mean_image = self.X.mean(axis=0)

        warn_kwargs(self, kwargs)

    def get_data(self, source):
        '''
        Fetch the data from source.
        '''

        raise NotImplementedError('Eswar todo')

    def next(self, batch_size=None):
        '''
        Iterate the data.
        '''

        if batch_size is None:
            batch_size = self.batch_size

        if self.pos == -1:
            self.reset()

            if not self.inf:
                raise StopIteration

        x = self.X[self.pos:self.pos+batch_size]

        self.pos += batch_size
        if self.pos + batch_size > self.n:
            self.pos = -1

        outs = OrderedDict()
        outs[self.name] = x

        return outs

    def save_images(self, x, imgfile):
        '''
        Save images for visualization.
        '''
        raise NotImplementedError('TODO')
