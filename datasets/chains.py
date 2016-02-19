'''
Chains dataset.
'''

from collections import OrderedDict
import numpy as np
from os import path
from progressbar import ProgressBar
import random
import theano
from theano import tensor as T
import time

from . import Dataset
from caltech import CALTECH
from cifar import CIFAR
from euclidean import Euclidean
from mnist import MNIST
from uci import UCI
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

def load_data(dataset=None,
              train_batch_size=None,
              valid_batch_size=None,
              test_batch_size=None,
              **dataset_args):

    from caltech import CALTECH
    from cifar import CIFAR
    from mnist import MNIST
    from uci import UCI

    if dataset == 'mnist':
        C = MNIST
    elif dataset == 'cifar':
        C = CIFAR
    elif dataset == 'caltech':
        C = CALTECH
    elif dataset == 'uci':
        C = UCI
    elif dataset == 'euclidean':
        C = Euclidean

    if train_batch_size is not None:
        train = Chains(C, batch_size=train_batch_size,
                  mode='train',
                  inf=False,
                  **dataset_args)
    else:
        train = None
    if valid_batch_size is not None:
        valid = Chains(C, batch_size=valid_batch_size,
                  mode='valid',
                  inf=False,
                  **dataset_args)
    else:
        valid = None
    if test_batch_size is not None:
        test = Chains(C, batch_size=test_batch_size,
                 mode='test',
                 inf=False,
                 **dataset_args)
    else:
        test = None

    return train, valid, test


class Chains(object):
    def __init__(self, D, batch_size=10,
                 window=20, l_chain=5000,
                 chain_stride=None, chain_noise=0.,
                 trim_end=0, out_path=None, **kwargs):

        if isinstance(D, Dataset):
            self.dataset = D
        else:
            self.dataset = D(batch_size=batch_size, shuffle=False, **kwargs)
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
        self.X = None
        self.P = None
        self.H = None
        self.C = None
        init_rngs(self, **kwargs)

    def set_chainer(self, chainer):
        self.chainer = chainer

    def build_chain(self, trim_end=0, condition_on=None):
        if self.chainer is None:
            raise ValueError('Chainer not set. Use `set_chainer` method.')

        data_pos = self.dataset.pos
        if data_pos == -1:
            self.dataset.reset()
            raise StopIteration

        chain_dict = self.chainer.build_data_chain(
            self.dataset, l_chain=self.l_chain, c=condition_on)

        self.X = chain_dict['x_chain']
        self.P = chain_dict['p_chain']
        self.H = chain_dict['h_chain']

        if self.out_path is not None:
            self.dataset.save_images(x,
                path.join(self.out_path, '%s_chain_%d.png'
                          % (self.dataset.mode, data_pos)), x_limit=200)

        if trim_end:
            print 'Trimming %d' % trim_end
            self.X = self.X[:-trim_end]
            self.P = self.P[:-trim_end]
            self.H = self.H[:-trim_end]

            if self.out_path is not None:
                self.dataset.save_images(x,
                    path.join(self.out_path, '%s_chain_%d_trimmed.png'
                              % (self.dataset.mode, data_pos)), x_limit=200)

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

    def get_batches(self, cs):
        x = []
        p = []
        h = []
        for c in cs:
            x.append(self.X[c])
            p.append(self.P[c])
            h.append(self.H[c])

        x = np.array(x).astype(floatX)
        p = np.array(p).astype(floatX)
        h = np.array(h).astype(floatX)

        return x, p, h

    def reset(self):
        self.dataset.reset()
        self.cpos = -1

    def randomize(self):
        rnd_idx = np.random.permutation(np.arange(0, len(self.C), 1))
        self.C = [self.C[i] for i in rnd_idx]

    def next(self, batch_size=None, l_chain=None, condition_on=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.cpos == -1:
            self.cpos = 0
            self.build_chain(trim_end=self.trim_end, condition_on=condition_on)
            window = min(self.window, self.X.shape[0])
            self.C = []
            for i in xrange(0, self.X.shape[0] - window + 1, self.chain_stride):
                self.C.append(range(i, i + window))
            self.randomize()

        try:
            cs = [self.C[self.cpos + b] for b in range(batch_size)]
        except:
            assert False, (len(self.C), self.cpos, batch_size)
        x, p, h = self.get_batches(cs)

        if self.cpos + 2 * self.batch_size >= len(self.C):
            self.cpos = -1
        else:
            self.cpos += self.batch_size

        rval = OrderedDict(
            x=x,
            p=p,
            h=h
        )

        return rval

    def next_simple(self, batch_size=None):
        x = self.dataset.next(batch_size=batch_size)
        return x