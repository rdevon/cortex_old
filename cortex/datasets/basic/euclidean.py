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
import scipy

from .. import Dataset
from ...utils import floatX, intX, pi
from ...utils.tools import init_rngs


class Euclidean(Dataset):
    def __init__(self, dims=2, n_samples=10000, **kwargs):
        super(Euclidean, self).__init__(**kwargs)
        init_rngs(self, **kwargs)

        self.collection = None
        self.X = self.get_data(n_samples, dims)
        self.make_fibrous()

        self.n = self.X.shape[0]

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

    def make_circle(self, r=0.3, G=0.05):
        for k in xrange(10):
            x = self.X[:, 0] - 0.5
            y = self.X[:, 1] - 0.5
            alpha = np.sqrt(x ** 2 + y ** 2) / r
            d = np.array([x * (1 - alpha), y * (1 - alpha)]).astype(floatX).T
            f = G * d
            self.X += f
            self.X = np.clip(self.X, 0, 1)

    def make_spiral(self, r=0.25, G=0.0001):
        for k in range(10):
            x = self.X[:, 0] - 0.5
            y = self.X[:, 1] - 0.5
            theta = np.arctan2(x, y)
            ds = [r * (i + theta / (2 * np.pi)) for i in range(int(1 / r))]
            alphas = [np.sqrt(x ** 2 + y ** 2) / d for d in ds]
            for alpha in alphas:
                d = np.concatenate([(x * (1 - alpha))[:, None], (y * (1 - alpha))[:, None]], axis=1)
                f = -G * d / (d ** 2).sum(axis=1, keepdims=True)
                self.X += f
            self.X = np.clip(self.X, 0, 1)

        rs = np.arange(0, 0.7, 0.001)
        theta = 2 * np.pi * rs / r
        y = rs * np.sin(theta) + 0.5
        x = -rs * np.cos(theta) + 0.5
        spiral = zip(x, y)
        self.collection = matplotlib.collections.LineCollection([spiral], colors='k')

    def make_ex(self):
        x = self.rng.normal(loc=0.5, scale=0.05, size=self.X.shape).astype(floatX)
        t1 = self.rng.uniform(low=-0.5, high=0.5, size=(self.X.shape[0] // 2,)).astype(floatX)
        t2 = self.rng.uniform(low=-0.5, high=0.5, size=t1.shape).astype(floatX)
        self.X = np.concatenate([x[:x.shape[0]//2] + t1[:, None], x[x.shape[0]//2:] + t2[:, None] * np.array([1, -1])[None, :]]).astype(floatX)

        self.collection = matplotlib.collections.LineCollection([[(0, 0), (1, 1)], [(0, 1), (1, 0)]], colors='k')

    def make_modes(self, r=0.3, N=5, G=0.01):
        modes = [2 * np.pi * n / N for n in range(N)]
        self.X = np.concatenate([self.rng.normal(
            loc=0.5, scale=0.05, size=(self.X.shape[0] // N, self.X.shape[1])).astype(floatX)
                                 + np.array([(r * np.cos(mode)), (r * np.sin(mode))]).astype(floatX)[None, :]
                                 for mode in modes])

    def make_bullseye(self, r=0.3, G=0.08):
        self.make_circle(r=r, G=G)
        self.X = np.concatenate(
            [self.X,
             self.rng.normal(loc=0.5,
                             scale=0.05,
                             size=(self.X.shape[0] // 10,
                                   self.X.shape[1]))]).astype(floatX)

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

    def save_images(self, X, imgfile, density=False):
        ax = plt.axes()
        x = X[:, 0]
        y = X[:, 1]
        if density:
            xy = np.vstack([x,y])
            z = scipy.stats.gaussian_kde(xy)(xy)
            ax.scatter(x, y, c=z, marker='o', edgecolor='')
        else:
            ax.scatter(x, y, marker='o', c=range(x.shape[0]),
                        cmap=plt.cm.coolwarm)

        if self.collection is not None:
            self.collection.set_transform(ax.transData)
            ax.add_collection(self.collection)


        ax.text(x[0], y[0], str('start'), transform=ax.transAxes)
        ax.axis([-0.2, 1.2, -0.2, 1.2])
        fig = plt.gcf()

        plt.savefig(imgfile)
        plt.close()