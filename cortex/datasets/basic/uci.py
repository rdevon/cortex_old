'''
Iterator for UCI dataset
'''

import h5py
import numpy as np

from ...utils import floatX
from ...utils.tools import (
    concatenate,
    init_rngs,
    rng_,
    scan
)

class UCI(object):
    def __init__(self, batch_size=100, source=None, mode='train', shuffle=True,
                 inf=False, name='uci', stop=None):

        if source is None:
            raise ValueError('No source file provided')
        print 'Loading {name} ({mode} from {source})'.format(
            name=name, mode=mode, source=source)

        X = self.get_data(source, mode)
        if stop is not None:
            X = X[:stop]
        self.n = X.shape[0]
        self.dims = dict()
        self.dims[name] = X.shape[1]
        self.acts = dict()
        self.acts[name] = 'T.nnet.sigmoid'

        self.shuffle = shuffle
        self.pos = 0
        self.bs = batch_size
        self.inf = inf
        self.next = self._next

        self.X = X
        self.mean_image = np.zeros((X.shape[1])).astype(floatX)

        if self.shuffle:
            self.randomize()

    def get_data(self, source, mode):
        with h5py.File(source, 'r') as f:
            X = f[mode]
            X = X[:X.shape[0]].astype(floatX)

        return X

    def __iter__(self):
        return self

    def randomize(self):
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.X = self.X[rnd_idx, :]

    def next(self):
        raise NotImplementedError()

    def reset(self):
        self.pos = 0
        if self.shuffle:
            self.randomize()

    def _next(self, batch_size=None):
        if batch_size is None:
            batch_size = self.bs

        if self.pos == -1:
            self.reset()

            if not self.inf:
                raise StopIteration

        x = self.X[self.pos:self.pos+batch_size]

        self.pos += batch_size
        if self.pos + batch_size > self.n:
            self.pos = -1

        return x, None

    def save_images(self, x, imgfile, transpose=False, x_limit=None):
        pass
