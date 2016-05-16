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
from utils import floatX
from utils.tools import (
    concatenate,
    init_rngs,
    rng_,
    scan
)


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


def extend_dataset(base_class):
    '''Function to extend any dataset class to store RNN hiddens.

    Args:
        base_class: Dataset class or subclass.

    Returns:
        DatasetWithHiddens subclass.

    Raises:
        ValueError if base_class is not a Dataset class.

    '''

    if not issubclass(base_class, Dataset):
        raise ValueError('%s must be a Dataset class' % base_class)

    class DatasetWithHiddens(base_class):
        '''Datset that stores RNN hidden states along with corresponding data.

        This dataset wraps a Dataset class, storing hidden states for each of the
        samples.

        Attributes:
            Hs: list of theano.shared. Stored hidden states for dataset.
            idx: indices for Hs. For randomization.
        '''
        def __init__(self, dim_hs, **dataset_args):
            super(DatasetWithHiddens, self).__init__(**dataset_args)
            self.Hs = [theano.shared(np.zeros((self.n, dim_h,)).astype(floatX))
                       for dim_h in dim_hs]
            self.idx = range(self.n)

        def randomize(self):
            rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
            self.X = self.X[rnd_idx, :]
            self.idx = [self.idx[i] for i in rnd_idx]

        def next(self, batch_size=None):
            if batch_size is None:
                batch_size = self.batch_size

            if self.pos == -1:
                self.reset()
                if not self.inf:
                    raise StopIteration

            hs = [H.get_value()[self.idx][self.pos:self.pos+batch_size]
                  for H in self.Hs]
            rval = super(DatasetWithHiddens, self).next(batch_size=batch_size)
            rval['hs'] = hs
            return rval

    return DatasetWithHiddens


class DChains(object):
    '''Chains dataset for Dijkstra's Chainer.

    Special Chain Dataset for use with Dijkstra Chainer class.

    '''
    def __init__(self, D, dim_hs, batch_size=10, build_batch=100, out_path=None,
                 **dataset_args):
        self.dataset = extend_dataset(D)(dim_hs, batch_size=batch_size, **dataset_args)
        self.chainer = None
        self.batch_size = batch_size
        self.build_batch = build_batch

        self.out_path = out_path
        self.save_images = self.dataset.save_images

        self.pos = -1
        self.X    = None
        self.M    = None
        self.Hs   = None

    def set_chainer(self, chainer):
        self.chainer = chainer

    def build_chain(self):
        '''Builds a chain with dataset iterator.

        Using owned Dataset class instance, call self.chainer.build_data_chain.
        Optionally, save images from chains and trims end.

        Raises:
            StopIteration: Finished iterating through self.dataset
        '''

        if self.chainer is None:
            raise ValueError('Chainer not set. Use `set_chainer` method.')

        # Iterate through dataset. If reach the end, raise StopIteration
        data_pos = self.dataset.pos
        if data_pos == -1:
            self.dataset.reset()
            raise StopIteration

        # Build the chain and set members.
        chain_dict = self.chainer.build_data_chain(
            self.dataset, self.batch_size, build_batch=self.build_batch)

        self.M = chain_dict['mask']
        self.X = chain_dict['x_chain']
        self.idx = chain_dict['i_chain']
        self.Hs = chain_dict['h_chains']

        # Save images
        if self.out_path is not None:
            self.dataset.save_images(x,
                path.join(self.out_path, '%s_chain_%d.png'
                          % (self.dataset.mode, data_pos)), x_limit=200)

    def reset(self):
        '''Reset the iterator.'''
        self.dataset.reset()
        self.pos = -1

    def next(self, batch_size=None):
        '''Draw next set of chains.'''
        if batch_size is None:
            batch_size = self.batch_size

        if self.pos == -1:
            self.pos = 0
            self.build_chain()

        x = self.X[:, self.pos:self.pos+batch_size]
        m = self.M[:, self.pos:self.pos+batch_size]
        idx = self.idx[:, self.pos:self.pos+batch_size]
        hs = [H[:, self.pos:self.pos+batch_size] for H in self.Hs]

        if self.pos + batch_size >= self.X.shape[0]:
            self.pos = -1
        else:
            self.pos += 1

        rval = OrderedDict(
            x=x,
            idx=idx,
            mask=m,
            hs=hs
        )

        return rval

    def next_simple(self, batch_size=None):
        x = self.dataset.next(batch_size=batch_size)
        return x


class Chains(object):
    '''Chains dataset.

    Manages chain formation and iteration for training.

    Attributes:
        dataset: Dataset instance.
        chainer: Chainer instance. Implements chaining operation.
        window: int. Window of chain to draw from during iteration.
        trim_end: int. Number of samples to trim from built chain.
        l_chain: int. Length of built chain.
        chain_noise: bool. Use noise in chain formation.
        out_path: str. Path for saving chain image files.
        save_images: method. Method for saving images pulled from self.dataset.

    '''
    def __init__(self, D, batch_size=10,
                 window=20, l_chain=5000,
                 chain_stride=None, chain_noise=0.,
                 trim_end=0, out_path=None, **dataset_args):

        if isinstance(D, Dataset):
            self.dataset = D
        else:
            self.dataset = D(batch_size=batch_size, **dataset_args)

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

        self.pos = -1
        self.X    = None
        self.P    = None
        self.Hs   = None
        self.C    = None
        init_rngs(self, **dataset_args)

    def set_chainer(self, chainer):
        self.chainer = chainer

    def build_chain(self, trim_end=0, condition_on=None):
        '''Builds a chain with dataset iterator.

        Using owned Dataset class instance, call self.chainer.build_data_chain.
        Optionally, save images from chains and trims end.

        Args:
            trim_end: int.
            condition_on: np.matrix. Optional.

        Returns:
            None

        Raises:
            StopIteration: Finished iterating through self.dataset

        '''

        if self.chainer is None:
            raise ValueError('Chainer not set. Use `set_chainer` method.')

        # Iterate through dataset. If reach the end, raise StopIteration
        data_pos = self.dataset.pos
        if data_pos == -1:
            self.dataset.reset()
            raise StopIteration

        # Build the chain and set members.
        chain_dict = self.chainer.build_data_chain(
            self.dataset, self.batch_size, l_chain=self.l_chain, c=condition_on)

        self.X = chain_dict['x_chain']
        self.P = chain_dict['p_chain']
        self.Hs = chain_dict['h_chain']


        # Save images
        if self.out_path is not None:
            self.dataset.save_images(x,
                path.join(self.out_path, '%s_chain_%d.png'
                          % (self.dataset.mode, data_pos)), x_limit=200)

        # Trim end
        if trim_end:
            print 'Trimming %d' % trim_end
            self.X = self.X[:-trim_end]
            self.P = self.P[:-trim_end]
            self.Hs = [H[:-trim_end] for H in self.Hs]

            if self.out_path is not None:
                self.dataset.save_images(x,
                    path.join(self.out_path, '%s_chain_%d_trimmed.png'
                              % (self.dataset.mode, data_pos)), x_limit=200)

    def get_labels(self, chains=None):
        '''Retrieve the labels from a chain.'''
        if chains is None:
            chains = [self.chain]
        y = []
        for chain in chains:
            y_ = []
            for c in chain:
                y_.append(self.dataset.Y[c])
            y.append(y_)
        return np.array(y).astype('float32')

    def get_batches(self, c):
        '''Get chain batches.'''
        hs = []
        x = self.X[c]
        p = self.P[c]
        for i in range(len(self.Hs)):
            hs.append(self.Hs[i][c])

        hs = np.array(hs).astype(floatX)

        return x, p, hs

    def reset(self):
        '''Reset the iterator.'''
        self.dataset.reset()
        self.pos = -1

    def randomize(self):
        '''Randomize the chains.'''
        rnd_idx = np.random.permutation(np.arange(0, len(self.C), 1))
        self.C = [self.C[i] for i in rnd_idx]

    def next(self, batch_size=None, l_chain=None, condition_on=None):
        '''Draw next set of chains.'''
        if batch_size is None:
            batch_size = self.batch_size

        if self.pos == -1:
            self.pos = 0
            self.build_chain(trim_end=self.trim_end, condition_on=condition_on)
            window = min(self.window, self.X.shape[0])
            self.C = []
            for i in xrange(0, self.X.shape[0] - window + 1, self.chain_stride):
                self.C.append(range(i, i + window))
            self.randomize()

        c = self.C[self.pos]

        x, p, hs = self.get_batches(c)

        if self.pos + 1 >= len(self.C):
            self.pos = -1
        else:
            self.pos += 1

        rval = OrderedDict(
            x=x,
            p=p,
            hs=hs
        )

        return rval

    def next_simple(self, batch_size=None):
        x = self.dataset.next(batch_size=batch_size)
        return x