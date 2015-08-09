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
from vis_utils import tile_raster_images


def reshape_image(img, shape):
    img.thumbnail(shape, PIL.Image.BILINEAR)
    new_img = PIL.Image.new('L', shape)
    offset_x = max((shape[0] - img.size[0]) / 2, 0)
    offset_y = max((shape[1] - img.size[1]) / 2, 0)
    offset_tuple = (offset_x, offset_y)
    new_img.paste(img, offset_tuple)
    return new_img


class Horses(object):
    def __init__(self, batch_size=10, source=None,
                 inf=False, chain_length=20, chains_build_batch=100,
                 stop=None, out_path=None, dims=None, chain_stride=None):
        # load MNIST
        assert source is not None

        self.dims = dims

        data = []
        for f in glob(path.join(path.abspath(source), '*.png')):
            img = PIL.Image.open(f)
            if self.dims is None:
                self.dims = img.size
            img = reshape_image(img, self.dims)
            data.append(np.array(img))

        X = np.array(data).astype('float32')
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        X = (X - X.min()) / float(X.max() - X.min())

        self.f_energy = None
        self.chain_length = chain_length
        self.out_path = out_path
        if chain_stride is None:
            self.chain_stride = self.chain_length
        else:
            self.chain_stride = chain_stride

        self.n = X.shape[0]
        if stop is not None:
            X = X[:stop]
            self.n = stop
        self.dim = X.shape[1]

        self.pos = 0
        self.bs = batch_size
        self.inf = inf
        self.X = X
        self.chain_pos = 0
        self.chains = [[] for _ in xrange(self.bs)]
        self.chains_build_batch = chains_build_batch
        self.next = self._next_model_chains_big

        # randomize
        print 'Shuffling mnist'
        self.randomize()

    def __iter__(self):
        return self

    def set_f_energy(self, f_energy_fn, model):
        self.f_energy = f_energy_fn(model)
        self.dim_h = model.dim_h

    def randomize(self):
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.X = self.X[rnd_idx, :]

    def next(self):
        raise NotImplementedError()

    def _load_chains(self, chains=None):
        if chains is None:
            chains = self.chains

        x = np.zeros((len(chains[0]), len(chains), self.dim)).astype('float32')
        for i, c in enumerate(chains):
            x[:, i] = self.X[c]
        return x

    def _build_chains(self, x_p=None, h_p=None):
        print 'Resetting chains'
        self.randomize()

        n_chains = len(self.chains)
        n_samples = self.chains_build_batch
        chain_idx = [range(self.n) for _ in xrange(n_chains)]
        for c in xrange(n_chains):
            rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
            chain_idx[c] = [chain_idx[c][i] for i in rnd_idx]

        counts = [[1 for _ in xrange(self.n)] for _ in xrange(n_chains)]
        n = self.n

        pos = 0
        while n > 0:
            stdout.write('\r%d         ' % n); stdout.flush()
            x_idx = []

            for c in xrange(n_chains):
                idx = [j for j in chain_idx[c] if counts[c][j] == 1]
                x_idx.append([idx[i] for i in range(pos, pos + min(n_samples, n))])

            uniques = np.unique(x_idx).tolist()
            x = self.X[uniques]
            wheres = [dict((i, j) for i, j in enumerate(uniques) if j in x_idx[c])
                for c in xrange(n_chains)]

            if n == self.n:
                picked_idx = [random.choice(idx) for idx in x_idx]
            else:
                assert x_p is not None
                energies, _, h_p = self.f_energy(x, x_p, h_p)
                picked_idx = []
                for c in xrange(n_chains):
                    idx = wheres[c].keys()
                    chain_energies = energies[c, idx]
                    i = np.argmin(chain_energies)
                    j = wheres[c][idx[i]]
                    picked_idx.append(j)

            for c in xrange(n_chains):
                j = picked_idx[c]
                counts[c][j] = 0
                chain = self.chains[c]
                self.chains[c].append(j)

            x_p = self.X[picked_idx]
            n -= 1

            if pos + n_samples > n:
                pos = 0
                for c in xrange(n_chains):
                    rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
                    chain_idx[c] = [chain_idx[c][i] for i in rnd_idx]

    def _next_model_chains_big(self):
        assert self.f_energy is not None

        cpos = self.chain_pos
        if cpos == -1 or len(self.chains[0]) == 0:
            self.chain_pos = 0
            cpos = self.chain_pos
            self.chains = [[] for _ in xrange(self.bs)]
            x_p = None
            h_p = np.random.normal(loc=0, scale=1, size=(self.bs, self.dim_h)).astype('float32')
            self._build_chains(x_p=x_p, h_p=h_p)
            if self.out_path:
                self.save_images(self._load_chains(),
                                 path.join(self.out_path,'training_chain.png'))

        chains = [[chain[j] for j in xrange(cpos, cpos+self.chain_length)]
            for chain in self.chains]

        x = self._load_chains(chains=chains)

        if cpos + self.chain_stride + self.chain_length > self.n:
            self.chain_pos = -1
        else:
            self.chain_pos += self.chain_stride

        return x, None

    def save_images(self, x, imgfile, transpose=False):
        if len(x.shape) == 2:
            x = x.reshape((x.shape[0], 1, x.shape[1]))
        tshape = x.shape[0], x.shape[1]
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        image = self.show(x.T, tshape, transpose=transpose)
        #print 'Saving to ', imgfile
        image.save(imgfile)

    def show(self, image, tshape, transpose=False):
        fshape = (self.dims[1], self.dims[0])
        if transpose:
            X = image
        else:
            X = image.T

        return PIL.Image.fromarray(tile_raster_images(
            X=X, img_shape=fshape, tile_shape=tshape,
            tile_spacing=(1, 1)))

    def translate(self, x):
        return x


class SimpleHorses(object):
    def __init__(self, batch_size=10, source=None, inf=False, stop=None,
                 dims=None, ):
        # load MNIST
        assert source is not None

        self.dims = dims

        data = []
        for f in glob(path.join(path.abspath(source), '*.png')):
            img = PIL.Image.open(f)
            if self.dims is None:
                self.dims = img.size
            img = reshape_image(img, self.dims)
            data.append(np.array(img))

        X = np.array(data)
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        X = (X - X.min()) / (X.max() - X.min())

        self.n = X.shape[0]
        if stop is not None:
            X = X[:stop]
            self.n = stop
        self.dim = X.shape[1]

        self.pos = 0
        self.bs = batch_size
        self.inf = inf
        self.X = X
        self.next = self._next

        # randomize
        print 'Shuffling mnist'
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

        cpos = self.pos
        if cpos + batch_size > self.n:
            # reset
            self.pos = 0
            cpos = self.pos
            if self.shuffle:
                self.randomize()
            if not self.inf:
                raise StopIteration

        x = self.X[cpos:cpos+batch_size]

        self.pos += batch_size

        return x, None

    def save_images(self, x, imgfile, transpose=False):
        if len(x.shape) == 2:
            x = x.reshape((x.shape[0], 1, x.shape[1]))
        tshape = x.shape[0], x.shape[1]
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        image = self.show(x.T, tshape, transpose=transpose)
        #print 'Saving to ', imgfile
        image.save(imgfile)

    def show(self, image, tshape, transpose=False):
        fshape = (self.dims[1], self.dims[0])
        if transpose:
            X = image
        else:
            X = image.T

        return PIL.Image.fromarray(tile_raster_images(
            X=X, img_shape=fshape, tile_shape=tshape,
            tile_spacing=(1, 1)))
