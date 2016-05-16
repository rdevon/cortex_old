'''
MNIST dataset
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

from .. import BasicDataset, Dataset
from ...utils.tools import (
    concatenate,
    init_rngs,
    resolve_path,
    rng_,
    scan
)
from ...utils.vis_utils import tile_raster_images


class MNIST(BasicDataset):
    def __init__(self, source=None, restrict_digits=None, mode='train',
                 binarize=False, name='mnist',
                 out_path=None, **kwargs):
        source = resolve_path(source)

        if source is None:
            raise ValueError('No source file provided')
        print 'Loading {name} ({mode}) from {source}'.format(
            name=name, mode=mode, source=source)

        X, Y = self.get_data(source, mode)

        if restrict_digits is not None:
            X = np.array([x for i, x in enumerate(X) if Y[i] in restrict_digits])
            Y = np.array([y for i, y in enumerate(Y) if Y[i] in restrict_digits])

        data = {name: X, 'label': Y}
        distributions = {name: 'binomial', 'label': 'multinomial'}

        super(MNIST, self).__init__(data, distributions=distributions,
                                    name=name, mode=mode, **kwargs)

        self.image_shape = (28, 28)
        self.out_path = out_path

        if binarize:
            self.data[name] = rng_.binomial(
                p=self.data[name], size=self.data[name].shape, n=1).astype('float32')

        self.mean_image = self.data[name].mean(axis=0)

        if self.shuffle:
            self.randomize()

    def get_data(self, source, mode):
        with gzip.open(source, 'rb') as f:
            x = cPickle.load(f)

        if mode == 'train':
            X = np.float32(x[0][0])
            Y = np.float32(x[0][1])
        elif mode == 'valid':
            X = np.float32(x[1][0])
            Y = np.float32(x[1][1])
        elif mode == 'test':
            X = np.float32(x[2][0])
            Y = np.float32(x[2][1])
        else:
            raise ValueError()

        return X, Y

    def save_images(self, x, imgfile, transpose=False, x_limit=None):
        if len(x.shape) == 2:
            x = x.reshape((x.shape[0], 1, x.shape[1]))

        if x_limit is not None and x.shape[0] > x_limit:
            x = np.concatenate([x, np.zeros((x_limit - x.shape[0] % x_limit,
                                             x.shape[1],
                                             x.shape[2])).astype('float32')],
                axis=0)
            x = x.reshape((x_limit, x.shape[0] * x.shape[1] // x_limit, x.shape[2]))

        if transpose:
            x = x.reshape((x.shape[0], x.shape[1], self.image_shape[0], self.image_shape[1]))
            x = x.transpose(0, 1, 3, 2)
            x = x.reshape((x.shape[0], x.shape[1], self.image_shape[0] * self.image_shape[1]))

        tshape = x.shape[0], x.shape[1]
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        image = self.show(x.T, tshape)
        image.save(imgfile)

    def show(self, image, tshape):
        fshape = self.image_shape
        X = image.T

        return PIL.Image.fromarray(tile_raster_images(
            X=X, img_shape=fshape, tile_shape=tshape,
            tile_spacing=(1, 1)))

    def translate(self, x):
        return x
