'''
Module for CIFAR dataset.
'''

from collections import OrderedDict
import cPickle
import gzip
import numpy as np
from os import path

from . import TwoDImageDataset
from ...utils.tools import resolve_path


class CIFAR(TwoDImageDataset):
    '''CIFAR dataset.

    '''
    def __init__(self, source=None, restrict_classes=None, mode='train',
                 name='cifar', greyscale=False, **kwargs):
        if source is None:
            raise TypeError('No source file provided')

        logger.info('Loading {name} ({mode}) from {source}'.format(
            name=name, mode=mode, source=source))

        source = resolve_path(source)

        X, Y = self.get_data(source, mode, greyscale=greyscale)

        if restrict_classes is not None:
            X = np.array(
                [x for i, x in enumerate(X) if Y[i] in restrict_classes])
            Y = np.array(
                [y for i, y in enumerate(Y) if Y[i] in restrict_classes])

        data = {'input': X, 'labels': Y}
        distributions = {'input': 'gaussian', 'labels': 'multinomial'}

        super(CIFAR, self).__init__(data, distributions=distributions,
                                    name=name, mode=mode, image_shape=(28, 28),
                                    **kwargs)

    def get_data(self, source, mode, greyscale=False):
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

        assert False, X.shape

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

_classes = {'CIFAR': CIFAR}