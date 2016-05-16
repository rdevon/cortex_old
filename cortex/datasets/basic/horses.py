import cPickle
from glob import glob
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
import traceback

from .mnist import MNIST
from ...utils.vis_utils import tile_raster_images


def reshape_image(img, shape, crop_image=True):
    if crop_image:
        bbox = img.getbbox()
        img = img.crop(bbox)

    img.thumbnail(shape, PIL.Image.BILINEAR)
    new_img = PIL.Image.new('L', shape)
    offset_x = max((shape[0] - img.size[0]) / 2, 0)
    offset_y = max((shape[1] - img.size[1]) / 2, 0)
    offset_tuple = (offset_x, offset_y)
    new_img.paste(img, offset_tuple)
    return new_img

class Horses(object):
    def __init__(self, batch_size=10, mode='train',
                 source=None, inf=False, stop=None, shuffle=True,
                 image_shape=None, out_path=None):

        assert source is not None
        print 'Loading horses ({mode})'.format(mode=mode)

        self.image_shape = image_shape

        data = []
        for f in glob(path.join(path.abspath(source), '*.png')):
            img = PIL.Image.open(f)
            if self.image_shape is None:
                self.image_shape = img.size
            img = reshape_image(img, self.image_shape)
            data.append(np.array(img))

        self.image_shape = self.image_shape[1], self.image_shape[0]

        X = np.array(data)
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        X = (X - X.min()) / (X.max() - X.min())

        self.n = X.shape[0]
        if stop is not None:
            X = X[:stop]
            self.n = stop
        self.dims = dict(horses=X.shape[1])
        self.acts = dict(horses='T.nnet.sigmoid')

        self.shuffle = shuffle
        self.pos = 0
        self.bs = batch_size
        self.inf = inf
        self.X = X
        self.next = self._next
        self.mean_image = self.X.mean(axis=0)

        if self.shuffle:
            print 'Shuffling horses'
            self.randomize()

    def __iter__(self):
        return self

    def randomize(self):
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.X = self.X[rnd_idx, :]

    def next(self):
        raise NotImplementedError()

    def _next(self, batch_size=None):
        if batch_size is None:
            batch_size = self.bs

        if self.pos == -1:
            self.reset()

            if not self.inf:
                raise StopIteration

        x = self.X[self.pos:self.pos+batch_size]

        self.pos += batch_size

        return x, None

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
        fshape = self.dims
        if transpose:
            X = image
        else:
            X = image.T

        return PIL.Image.fromarray(tile_raster_images(
            X=X, img_shape=fshape, tile_shape=tshape,
            tile_spacing=(1, 1)))
