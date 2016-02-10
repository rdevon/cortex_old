'''
Simple dataset with random points arrange in a space.
'''

from collections import OrderedDict
import numpy as np

from . import Dataset
from utils.tools import (
    floatX,
    init_rngs
)


class Euclidean(Dataset):
    def __init__(self, dims=2, n_samples=10000, **kwargs):
        super(Euclidean, self).__init__(**kwargs)
        init_rngs(self, **kwargs)

        self.X = self.get_data(n_samples, dims)
        self.n = self.X.shape[0]
        self.dims = dict()
        self.dims[self.name] = dims
        self.distributions = dict()
        self.distributions[self.name] = 'continuous_binomial'

        self.mean_image = self.X.mean(axis=0)

    def randomize(self):
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.X = self.X[rnd_idx, :]

    def get_data(self, n_points, dims):
        x = self.rng.uniform(size=(n_points, dims)).astype(floatX)
        return x

    def next(self, batch_size=None):
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
