'''
Module for cifar
'''

from collections import OrderedDict
import cPickle
import gzip
import multiprocessing as mp
import numpy as np
from os import path
import PIL
import random
import sys
from sys import stdout
import theano
from theano import tensor as T
import time
import traceback

from ...utils import floatX
from ...utils.tools import (
    concatenate,
    init_rngs,
    resolve_path,
    rng_,
    scan
)
from ...utils.vis_utils import tile_raster_images


def get_iter(inf=False, batch_size=128):
    return mnist_iterator(inf=inf, batch_size=batch_size)

class CIFAR(object):
    def __init__(self, batch_size=128, source=None,
                 restrict_digits=None, mode='train', shuffle=True, inf=False,
                 stop=None, out_path=None):
        source = resolve_path(source)
        self.name = 'cifar'

        X, Y = self.get_data(source, mode)
        self.mode = mode

        self.image_shape = (32, 32)
        self.out_path = out_path

        if restrict_digits is None:
            n_classes = 10
        else:
            n_classes = len(restrict_digits)

        O = np.zeros((X.shape[0], n_classes), dtype='float32')

        if restrict_digits is None:
            for idx in xrange(X.shape[0]):
                O[idx, Y[idx]] = 1.;
        else:
            print 'Restricting to classes %s' % restrict_digits
            new_X = []
            i = 0
            for j in xrange(X.shape[0]):
                if Y[j] in restrict_digits:
                    new_X.append(X[j])
                    c_idx = restrict_digits.index(Y[j])
                    O[i, c_idx] = 1.;
                    i += 1
            X = np.float32(new_X)

        if stop is not None:
            X = X[:stop]

        self.n = X.shape[0]
        print 'Data shape: %d x %d' % X.shape

        self.dims = dict(cifar=X.shape[1], label=len(np.unique(Y)))
        self.distributions = dict(cifar='gaussian', label='multinomial')

        self.shuffle = shuffle
        self.pos = 0
        self.bs = batch_size
        self.inf = inf
        self.next = self._next
        self.X = X
        self.O = O

        self.mean_image = self.X.mean(axis=0)
        self.X -= self.mean_image
        self.X /= self.X.std(axis=0)

        if self.shuffle:
            self.randomize()

    def get_data(self, source, mode, greyscale=True):
        if not greyscale:
            raise NotImplementedError()
        if source is None:
            raise ValueError('No source file provided')
        print 'Loading CIFAR-10 ({mode})'.format(mode=mode)

        X = []
        Y = []

        if mode == 'train':
            for i in xrange(1, 5):
                with open(path.join(source, 'data_batch_%d' % i)) as f:
                    d = cPickle.load(f)
                    X.append(d['data'])
                    Y.append(d['labels'])
        elif mode == 'valid':
            with open(path.join(source, 'data_batch_5')) as f:
                d = cPickle.load(f)
                X.append(d['data'])
                Y.append(d['labels'])
        elif mode == 'test':
            with open(path.join(source, 'test_batch')) as f:
                d = cPickle.load(f)
                X.append(d['data'])
                Y.append(d['labels'])
        else:
            raise ValueError()

        X = np.concatenate(X)
        Y = np.concatenate(Y)

        if greyscale:
            div = X.shape[1] // 3
            X_r = X[:, :div]
            X_b = X[:, div:2*div]
            X_g = X[:, 2*div:]
            X = (X_r + X_b + X_g) / 3.0

        X = X.astype(floatX)
        X = X / float(X.max())
        X = (X - X.mean(axis=0))# / X.std(axis=0)

        return X, Y

    def __iter__(self):
        return self

    def randomize(self):
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.X = self.X[rnd_idx, :]
        self.O = self.O[rnd_idx, :]

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
        y = self.O[self.pos:self.pos+batch_size]

        self.pos += batch_size
        if self.pos + batch_size > self.n:
            self.pos = -1

        return OrderedDict(cifar=x, labels=y)

    def save_images(self, x, imgfile, transpose=False, x_limit=None):
        if len(x.shape) == 2:
            x = x.reshape((x.shape[0], 1, x.shape[1]))

        if x_limit is not None and x.shape[0] > x_limit:
            x = np.concatenate([x, np.zeros((x_limit - x.shape[0] % x_limit,
                                             x.shape[1],
                                             x.shape[2])).astype('float32')],
                axis=0)
            x = x.reshape((x_limit, x.shape[0] * x.shape[1] // x_limit, x.shape[2]))

        tshape = x.shape[0], x.shape[1]
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        image = self.show(x.T, tshape, transpose=transpose)
        image.save(imgfile)

    def show(self, image, tshape, transpose=False):
        fshape = self.image_shape
        if transpose:
            X = image
        else:
            X = image.T

        return PIL.Image.fromarray(tile_raster_images(
            X=X, img_shape=fshape, tile_shape=tshape,
            tile_spacing=(1, 1)))

    def translate(self, x):
        return x