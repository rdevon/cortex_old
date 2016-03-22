'''
SNP dataset class.
'''

from collections import OrderedDict
import numpy as np
from scipy.io import loadmat
from . import Dataset
from utils.tools import (
    floatX,
    warn_kwargs
)


class SNP(Dataset):
    '''SNP dataset class.

    Currently only handled continuous preprocessed data.
    Discrete data TODO
    '''
    def __init__(self, source=None, name='snp', idx=None, **kwargs):
        '''Initialize the SNP dataset.

        Arguments:
            source: str. Path to source file.
            name: str. ID of dataset.
            idx: (Optional) list. List of indices for train/test/validation
                split.
        '''
        kwargs = super(SNP, self).__init__(name=name, **kwargs)

        if source is None:
            raise ValueError('No source provided')

        # Fetch SNP data from "source"
        self.get_data(source)
        # Reference for the dimension of the dataset. A dict is used for
        # multimodal data (e.g., mri and labels)
        self.dims = {self.name: self.X.shape[1],'labels': self.Y.shape[1]}

        # This is reference for models to decide how the data should be modelled
        # E.g. with a binomial or gaussian variable
        self.distributions = {self.name: 'gaussian', 'labels': 'multinomial'}

        self.mean_image = self.X.mean(axis=0)

        if idx is not None:
            self.X = self.X[idx]
            self.Y = self.Y[idx]

        self.n = self.X.shape[0]

    def get_data(self, source):
        '''Fetch the data from source.

        TODO.

        Arguments:
            source: TODO
        '''
        from utils.tools import get_paths
        data_path = get_paths()['$snp_data']
        print('Loading genetic data from %s' % data_path)
        X = loadmat(data_path + source['snp'])
        Y = loadmat(data_path + source['labels'])
        self.X = X[X.keys()[2]]
        self.Y = Y[Y.keys()[0]]

    def randomize(self):
        '''Randomize the dataset.'''
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.X = self.X[rnd_idx,:]
        self.Y = self.Y[rnd_idx]

    def reset(self):
        '''Reset the iterator'''
        self.pos = 0
        if self.shuffle:
            self.randomize()

    def next(self, batch_size=None):
        '''Iterate the data.'''

        if batch_size is None:
            batch_size = self.batch_size

        if self.pos == -1:
            self.reset()
            raise StopIteration

        x = self.X[self.pos:self.pos + batch_size]
        y = self.Y[self.pos:self.pos + batch_size]

        rval = OrderedDict(pos=self.pos)

        if self.pos + batch_size > self.n:
            self.pos = -1
        else:
            self.pos += batch_size

        rval[self.name] = x
        rval['labels'] = y

        return rval
