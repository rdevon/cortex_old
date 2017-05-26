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

from .. import BasicDataset
from ...utils import floatX, intX, pi, _rng


class Euclidean(BasicDataset):

    def __init__(self, dims=2, n_samples=10000, method='fibrous', name=None,
                 method_args=None, mode='train', **kwargs):
        if name is None: name = method
        if method_args is None: method_args = dict()

        self.collection = None
        X = self.get_data(n_samples, dims)

        _method_dict = {
            'fibrous': self.make_fibrous,
            'circle': self.make_circle,
            'sprial': self.make_spiral,
            'X': self.make_ex,
            'modes': self.make_modes,
            'bullseye': self.make_bullseye
        }

        X = _method_dict[method](X, **method_args)

        data = {'input': X}
        distributions = {'input': 'gaussian'}

        super(Euclidean, self).__init__(data, distributions=distributions,
                                        name=name, mode=mode, **kwargs)

    def gravity(self, x, y, r=0.1, G=0.0001):
        d = np.sqrt(((y[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))

        d_cb = d ** 3
        d_r = y[:, None, :] - x[None, :, :]

        f = -(G * d_r / d_cb[:, :, None])
        c = (d < r).astype(intX)
        f = f * c[:, :, None]
        f = f.sum(axis=0)

        return f

    def make_circle(self, X, r=0.3, G=0.05):
        for k in xrange(10):
            x = X[:, 0] - 0.5
            y = X[:, 1] - 0.5
            alpha = np.sqrt(x ** 2 + y ** 2) / r
            d = np.array([x * (1 - alpha), y * (1 - alpha)]).astype(floatX).T
            f = G * d
            X += f
            X = np.clip(X, 0, 1)
        return X

    def make_spiral(self, X, r=0.25, G=0.0001):
        for k in range(10):
            x = X[:, 0] - 0.5
            y = X[:, 1] - 0.5
            theta = np.arctan2(x, y)
            ds = [r * (i + theta / (2 * np.pi)) for i in range(int(1 / r))]
            alphas = [np.sqrt(x ** 2 + y ** 2) / d for d in ds]
            for alpha in alphas:
                d = np.concatenate([(x * (1 - alpha))[:, None], (y * (1 - alpha))[:, None]], axis=1)
                f = -G * d / (d ** 2).sum(axis=1, keepdims=True)
                X += f
            X = np.clip(X, 0, 1)

        rs = np.arange(0, 0., 0.001)
        theta = 2 * np.pi * rs / r
        y = rs * np.sin(theta) + 0.5
        x = -rs * np.cos(theta) + 0.5
        spiral = zip(x, y)
        self.collection = matplotlib.collections.LineCollection([spiral], colors='k')
        return X

    def make_ex(self, X):
        x = _rng.normal(loc=0.5, scale=0.05, size=X.shape).astype(floatX)
        t1 = _rng.uniform(low=-0.5, high=0.5, size=(X.shape[0] // 2,)).astype(floatX)
        t2 = _rng.uniform(low=-0.5, high=0.5, size=t1.shape).astype(floatX)
        X = np.concatenate(
            [x[:x.shape[0]//2] + t1[:, None],
             x[x.shape[0]//2:] + t2[:, None] * np.array([1, -1])[None, :]]
            ).astype(floatX)

        self.collection = matplotlib.collections.LineCollection(
            [[(0, 0), (1, 1)], [(0, 1), (1, 0)]], colors='k')
        return X

    def make_modes(self, X, r=0.3, N=5, scale=0.03, placement='random'):
        modes = [2 * np.pi * n / N for n in range(N)]
        
        if placement == 'circle':
            displacements = [np.array(
                [(r * np.cos(mode)), (r * np.sin(mode))]).astype(floatX)
                             for mode in modes]
            
        elif placement == 'random':
            displacements = [np.array(
                [_rng.uniform(-0.5, 0.5), _rng.uniform(-0.5, 0.5)]).astype(floatX)
                             for mode in modes]
        X = np.concatenate([_rng.normal(
            loc=0., scale=scale,
            size=(X.shape[0] // N, X.shape[1])).astype(floatX) + d[None, :]
                            for d in displacements])
        return X

    def make_bullseye(self, X, r=0.3, G=0.08):
        self.make_circle(r=r, G=G)
        X = np.concatenate(
            [X, _rng.normal(
                loc=0.5, scale=0.05, size=(
                    X.shape[0] // 10, X.shape[1]))]).astype(floatX)
        return X

    def make_fibrous(self, X, n_points=40):
        y = _rng.uniform(size=(n_points, X.shape[1])).astype(floatX)

        for k in xrange(10):
            f = self.gravity(X, y)
            X += f
            X = np.clip(X, 0, 1)
        return X

    def get_data(self, n_points, dims):
        x = _rng.uniform(size=(n_points, dims)).astype(floatX)
        return x
    
    def viz(self, X=None, out_file=None, density=False):
        self.save_images(X, out_file, density=density)

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
        ax.axis([-.7, .7, -.7, .7])
        fig = plt.gcf()

        plt.savefig(imgfile)
        plt.close()

_classes = {'euclidean': Euclidean}