'''
Chains dataset.
'''

import numpy as np
from os import path
from progressbar import ProgressBar
import random
import theano
from theano import tensor as T
import time

from . import Dataset
from utils.tools import (
    concatenate,
    floatX,
    init_rngs,
    rng_,
    scan
)


def energy(x, x_p, h_p, model):
    params = model.get_sample_params()
    h, x_s, p = model.step_sample(h_p[None, :], x_p[None, :], *params)
    energy = model.neg_log_prob(x, p)
    return energy, x_s[0], h[0]

def distance(x, x_p, h_p):
    distance = (x - x_p[None, :]) ** 2
    distance = distance.sum(axis=1)
    return distance, x, h_p

def random_distance(x, x_p, h_p):
    raise NotImplementedError()
    distance = model.trng.uniform(size=(x.shape[0],), dtype=x_p.dtype)
    return distance, x, h_p


class Chains(object):
    def __init__(self, D, batch_size=10,
                 window=20, l_chain=5000,
                 chain_stride=None, chain_noise=0.,
                 trim_end=0, out_path=None, **kwargs):

        if isinstance(D, Dataset):
            self.dataset = D
        else:
            self.dataset = D(batch_size=batch_size, **kwargs)
        self.chainer = None

        self.batch_size = batch_size
        self.window = window
        self.trim_end = trim_end
        self.l_chain = min(l_chain, self.dataset.n)
        self.chain_noise = chain_noise

        self.out_path = out_path
        self.save_images = self.dataset.save_images

        if chain_stride is None:
            self.chain_stride = self.window
        else:
            self.chain_stride = chain_stride

        self.cpos = -1
        self.chain = []
        init_rngs(self, **kwargs)

    def set_chainer(self, chainer):
        self.chainer = chainer

    def build_chain(self, trim_end=0, condition_on=None):
        if self.chainer is None:
            raise ValueError('Chainer not set. Use `set_chainer` method.')

        data_pos = self.dataset.pos
        if data_pos == -1:
            dataset.reset
            raise StopIteration

        chain_dict = self.chainer.build_data_chain(
            self.dataset, l_chain=self.l_chain, c=condition_on)

        chain = chain_dict['idx']
        self.chain = [int(i) + data_pos for i in chain]

        if self.out_path is not None:
            self.dataset.save_images(chain_dict['x_chain'],
                path.join(self.out_path, '%s_chain_%d.png'
                          % (self.dataset.mode, data_pos)),
                x_limit=200)

        if trim_end:
            print 'Trimming %d' % trim_end
            self.chain = self.chain[:-trim_end]

            if self.out_path is not None:
                self.dataset.save_images(
                    self._load_chains(),
                    path.join(self.out_path,
                              '%s_chain_%d_trimmed.png'
                              % (self.dataset.mode, data_pos)),
                    x_limit=200)

    def load_chains(self, chains=None):
        if chains is None:
            chains = [self.chain]

        dim = self.dataset.dims[self.dataset.name]

        x = np.zeros((len(chains[0]), len(chains), dim)).astype(floatX)
        for i, c in enumerate(chains):
            x[:, i] = self.dataset.X[c]
        return x

    def get_labels(self, chains=None):
        if chains is None:
            chains = [self.chain]

        y = []
        for chain in chains:
            y_ = []
            for c in chain:
                y_.append(self.dataset.Y[c])
            y.append(y_)
        return np.array(y).astype('float32')

    def batch_chain(self, chain=None, batch_size=None):
        window = min(self.window, len(self.chain))

        if chain is None:
            chain = self.chain
        if batch_size is None:
            batch_size = self.batch_size

        chains = []

        for b in xrange(self.batch_size):
            p = self.chain_idx[b + self.cpos]
            chains.append([chain[j] for j in xrange(p, p + window)])

        return chains

    def reset(self):
        self.dataset.reset()
        self.cpos = -1

    def next(self, batch_size=None, l_chain=None, condition_on=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.cpos == -1:
            self.cpos = 0
            self.build_chain(trim_end=self.trim_end, condition_on=condition_on)
            window = min(self.window, len(self.chain))
            self.chain_idx = range(0, len(self.chain) - window + 1, self.chain_stride)
            random.shuffle(self.chain_idx)

        chain_batch = self.batch_chain(batch_size=batch_size)
        x = self.load_chains(chains=chain_batch)

        if self.cpos + self.batch_size >= len(self.chain_idx):
            self.cpos = -1
        else:
            self.cpos += self.batch_size

        return x

    def next_simple(self, batch_size=None):
        x = self.dataset.next(batch_size=batch_size)
        return x