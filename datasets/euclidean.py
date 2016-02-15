'''
Simple dataset with random points arrange in a space.
'''

from collections import OrderedDict
import matplotlib
from matplotlib import pylab as plt
import numpy as np
from progressbar import (
    Bar,
    ProgressBar,
    Timer
)
import random

from . import Dataset
from utils import floatX, intX
from utils.tools import init_rngs


class Euclidean(Dataset):
    def __init__(self, dims=2, n_samples=10000, **kwargs):
        super(Euclidean, self).__init__(**kwargs)
        init_rngs(self, **kwargs)

        self.X = self.get_data(n_samples, dims)
        self.n = self.X.shape[0]

        self.make_circle()
        self.dims = dict()
        self.dims[self.name] = dims
        self.distributions = dict()
        self.distributions[self.name] = 'gaussian'

        self.mean_image = self.X.mean(axis=0)

    def gravity(self, x, y, r=0.1, G=0.0001):
        d = np.sqrt(((y[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))

        d_cb = d ** 3
        d_r = y[:, None, :] - x[None, :, :]

        f = -(G * d_r / d_cb[:, :, None])
        c = (d < r).astype(intX)
        f = f * c[:, :, None]
        f = f.sum(axis=0)

        return f

    def make_circle(self, r=0.3, G=0.3):
        for k in xrange(10):
            x = self.X[:, 0] - 0.5
            y = self.X[:, 1] - 0.5
            alpha = np.sqrt(x ** 2 + y ** 2) / r
            d = np.array([x * (1 - alpha), y * (1 - alpha)]).astype(floatX).T
            f = G * d
            self.X += f
            self.X = np.clip(self.X, 0, 1)


    def make_fibrous(self, n_points=40):
        y = self.rng.uniform(size=(n_points, self.X.shape[1])).astype(floatX)

        for k in xrange(10):
            f = self.gravity(self.X, y)
            self.X += f
            self.X = np.clip(self.X, 0, 1)

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

    def save_images(self, x, imgfile):
        fig = plt.figure()
        plt.scatter(x[:, 0], x[:, 1], marker='o', c=range(x.shape[0]),
                    cmap=plt.cm.coolwarm)
        plt.text(x[0, 0], x[0, 1], str('start'))
        #for i, (x, y) in enumerate(zip(x[:, 0], x[:, 1])):
        #    plt.text(x, y, str(i), color='black', fontsize=12)
        plt.savefig(imgfile)
        plt.close()
