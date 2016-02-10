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

from . import Dataset
from utils.tools import (
    concatenate,
    init_rngs,
    rng_,
    scan
)
from utils.vis_utils import tile_raster_images


class MNIST(Dataset):
    def __init__(self, source=None, restrict_digits=None, mode='train',
                 binarize=False, name='mnist',
                 out_path=None, **kwargs):
        super(MNIST, self).__init__(name=name, **kwargs)

        if source is None:
            raise ValueError('No source file provided')
        print 'Loading {name} ({mode})'.format(name=name, mode=mode)

        X, Y = self.get_data(source, mode)
        self.mode = mode

        self.image_shape = (28, 28)
        self.out_path = out_path

        uniques = np.unique(Y).tolist()
        if restrict_digits is None:
            n_classes = len(uniques)
        else:
            n_classes = len(restrict_digits)

        O = np.zeros((X.shape[0], n_classes), dtype='float32')

        if restrict_digits is None:
            for idx in xrange(X.shape[0]):
                i = uniques.index(Y[idx])
                O[idx, i] = 1.;
        else:
            print 'Restricting to digits %s' % restrict_digits
            new_X = []
            i = 0
            for j in xrange(X.shape[0]):
                if Y[j] in restrict_digits:
                    new_X.append(X[j])
                    c_idx = restrict_digits.index(Y[j])
                    O[i, c_idx] = 1.;
                    i += 1
            X = new_X.astype(floatX)

        if self.stop is not None:
            X = X[:self.stop]
        self.n = X.shape[0]

        self.dims = dict(label=len(np.unique(Y)))
        self.dims[name] = X.shape[1]
        self.distributions = dict(label='binomial')
        self.distributions[name] = 'binomial'

        if binarize:
            X = rng_.binomial(p=X, size=X.shape, n=1).astype('float32')

        self.X = X
        self.O = O

        self.mean_image = self.X.mean(axis=0)

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

    def randomize(self):
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.X = self.X[rnd_idx, :]
        self.O = self.O[rnd_idx, :]

    def next(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.pos == -1:
            self.reset()

            if not self.inf:
                raise StopIteration

        x = self.X[self.pos:self.pos+batch_size]
        y = self.O[self.pos:self.pos+batch_size]

        self.pos += batch_size
        if self.pos + batch_size > self.n:
            self.pos = -1

        rval = OrderedDict()
        rval[self.name] = x
        rval['label'] = y

        return rval

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





class MNIST_Pieces(MNIST):
    def __init__(self, batch_size=1, source='/Users/devon/Data/mnist.pkl.gz',
                 restrict_digits=None, mode='train', shuffle=True,
                 width=5, stride=5,
                 stop=None, out_path=None, chain_stride=None):
        with gzip.open(source, 'rb') as f:
            x = cPickle.load(f)

        X, Y = self.get_data(x, mode)

        self.dims = (28, 28)
        self.f_energy = None
        self.stride = stride
        self.bs = batch_size

        self.out_path = out_path
        if chain_stride is None:
            self.chain_stride = self.chain_length
        else:
            self.chain_stride = chain_stride

        self.shuffle = shuffle

        X, O = self.process(X, Y, restrict_digits)

        if stop is not None:
            X = X[:stop]

        Y = self.split(X. self.width, self.stride)

        self.n, self.chain_length, self.dim = Y.shape
        self.chains = [[] for _ in xrange(self.bs)]
        self.pos = 0
        self.spos = 0

        self.X = X
        self.Y = Y
        self.O = O

        if self.shuffle:
            print 'Shuffling mnist'
            self.randomize()

    def split(self, X, width, stride):
        step_x = range(0, self.dims[0] - width, stride)
        step_y = range(0, self.dims[1] - width, stride)
        n_windows = len(step_x) * len(step_y)
        Y = np.zeros((X.shape[0], n_windows, width ** 2))
        for i, S in enumerate(X):
            for x in step_x:
                for y in step_y:
                    p = x + y * width
                    idx = [range(p + w * self.dims[0],
                                 p + w * self.dims[0] + self.width)
                           for w in range(width)]
                    idx = [i for s in idx for i in s]
                    Y[i, p] = S[idx]

        return Y

    def randomize(self):
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.X = self.X[rnd_idx, :]
        self.O = self.O[rnd_idx, :]
        self.Y = self.Y[rnd_idx, :, :]

    def _load_chains(self, chains=None):
        if chains is None:
            chains = self.chains

        l_chains = len(chains[0])
        n_chains = len(chains)
        x = np.zeros((l_chains, n_chains, self.dim)).astype('float32')
        for i, (c, ps) in enumerate(chains):
            x[:, i] = self.Y[c + self.pos, ps]
        return x

    def _build_chain(self, x_p=None, h_p=None):
        n_chains = len(self.chains)
        l_chain = self.chain_length
        n_samples = l_chain
        chain_idx = [range(l_chain) for _ in xrange(n_chains)]

        print('Resetting chains with length %d and %d samples per query.'
              % (l_chain, n_samples))

        for c in xrange(n_chains):
            rnd_idx = np.random.permutation(np.arange(0, l_chain, 1))
            chain_idx[c] = [chain_idx[c][i] for i in rnd_idx]

        counts = [[True for _ in xrange(l_chain)] for _ in xrange(n_chains)]
        n = l_chain

        while n > 0:
            stdout.write('\r%d         ' % n); stdout.flush()
            x_idx = []

            for c in xrange(n_chains):
                idx = [j for j in chain_idx[c] if counts[c][j]]
                x_idx.append(idx)

            x = np.array([self.Y[c + self.pos, js] for c, js in enumerate(x_idx)]).astype(floatX)

            if n == l_chain:
                picked_idx = [random.choice(idx) for idx in x_idx]
            else:
                assert x_p is not None
                energies, _, h_p = self.f_energy(x, x_p, h_p)
                picked_idx = []
                for c, js in enumerate(x_idx):
                    chain_energies = energies[c]
                    i = np.argmin(chain_energies)
                    j = js[i]
                    picked_idx.append(j)

            for c in xrange(n_chains):
                j = picked_idx[c]
                counts[c][j] = False
                self.chains[c].append(j)

            x_p = np.array([self.Y[c, j]
                            for c, j
                            in enumerate(picked_idx)]).astype(floatX)
            n -= 1

    def next(self, save_chain=False):
        assert self.f_energy is not None

        if self.pos == 0:
            self.randomize()

        batch_size = min(self.bs, self.n - self.pos)
        self.chains = [[] for _ in xrange(batch_size)]

        h_p = np.random.normal(loc=0, scale=1, size=(batch_size, self.dim_h)).astype('float32')
        self._build_chain(x_p=None, h_p=h_p)

        assert len(np.unique(self.chains[0])) == len(self.chains[0]), (len(np.unique(self.chains[0])), len(self.chains[0]))
        for i in self.chains[0]:
            assert i >= self.pos and i < self.pos + self.chain_length

        if self.out_path and save_chain:
            self.save_images(
                self._load_chains(),
                path.join(self.out_path, 'training_chain_%d.png' % self.pos),
                x_limit=200)

        x = self._load_chains()
        ps = [[p for p in chain] for chain in self.chains]
        coords = np.array([[p % self.width, p // self.width] for p in ps]).astype(floatX)

        self.pos += self.bs
        if self.pos >= self.n:
            self.pos = 0
            self.randomize()

        return x, coords

    def next_simple(self, batch_size=10):
        cpos = self.spos
        if cpos + batch_size > self.n:
            self.spos = 0
            cpos = self.spos
            if self.shuffle:
                self.randomize()

        x = self.X[cpos:cpos+batch_size]
        self.spos += batch_size

        return x

    def draw(self, ps):
        pallet = T.alloc(
            0., ps.shape[0], ps.shape[1],
            self.dims[0] * self.dims[1]
            ).astype(floatX)

        def step_draw(p, pallet):
            x = p[:, :-4]
            pos = p[:, -4:-2]
            x_pos = floor(pos[0])
            y_pos = floor(pos[1])

        seqs = [ps]
