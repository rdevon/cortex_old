'''
Module for fMRI data
'''

import numpy as np
import os
import random


def make_datasets():
    pass

def medfilt(x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i+1)] = x[j:]
        y[-j:, -(i+1)] = x[-1]
    return np.median(y, axis=1)

def make_one_hot(labels):
    unique_labels = np.unique(labels).tolist()
    n_labels = len(unique_labels)
    one_hot = np.zeros((len(labels), n_labels))
    for i, l in enumerate(labels):
        j = unique_labels.index(l)
        one_hot[i][j] = 1
    return one_hot.astype('float32')


class ICA_Loadings(object):
    def __init__(self, batch_size=100, source=None, label_mode='one_hot',
                 idx=None, shuffle=True, window=10, stride=3, end_mode='clip'):
        self.X, self.Y, self.L = self.load_data(source, idx)

        self.n_runs, self.n_subjects, self.t, self.dim = X.shape

        last_index = (self.t - window) - (self.t - window) % stride
        if end_mode == 'clip':
            self.X = self.X[:, :last_index]
        elif end_mode == 'pad':
            raise NotImplementedError()

        if self.Y.shape[2] < self.t:
            self.Y = np.lib.pad(self.Y, ((0, 0),
                (0, self.t - self.Y.shape[2]),
                (0, 0)), 'constant', constant_values=0)

        if self.t < self.Y.shape[2]:
            self.Y = self.Y[:, :, :self.t]

        assert self.L.shape[0] == self.n_subjects

        if label_mode == 'one_hot':
            self.L = make_one_hot(self.L)

        self.n_runs, self.n_stims, _ = self.Y.shape
        assert self.Y.shape[2] == self.t

        self.bs = batch_size
        self.shuffle = shuffle

        self.indices = []
        for r in xrange(self.n_runs):
            for i in xrange(self.n_subjects):
                self.indices += [(r, i, j)
                    for j in range(0, self.t - window + 1, stride)]

        self.n = len(self.indices)

        if self.shuffle:
            self.randomize

    def load_data(self, source, idx):
        tc_file = os.path.join(source, 'tcs.npy')
        stim_file = os.path.join(source, 'stims.npy')
        label_file = os.path.join(source, 'labels.npy')
        X = np.load(tc_file)
        Y = np.load(stim_file).transpose((2, 0, 1))
        L = np.load(label_file)

        if idx is not None:
            X = X[idx]
            Y = Y[idx]
            L = L[idx]

        return X, Y, L

    def randomize(self, ):
        random.shuffle(self.indices)

    def next(batch_size=None):
        if batch_size is None:
            batch_size = self.bs

        if self.pos == -1:
            self.randomize()
            self.pos = 0
            if not self.inf:
                raise StopIteration

        batch_size = min(batch_size, self.n - self.pos)

        indices = [self.indices[p]
                   for p in xrange(self.pos, self.pos + batch_size)]

        x = np.zeros((batch_size, self.X.shape[2], self.X.shape[3])).astype('float32')
        y = np.zeros((batch_size, self.Y.shape[1], self.Y.shape[2])).astype('int64')
        l = np.zeros((batch_size,)).astype('int64')

        window = self.X.shape[2]

        for b, (r, i, j) in enumerate(indices):
            x[i] = self.X[r, i, j:j+window]
            y[i] = self.Y[r, :, j:j+window]
            l[i] = self.L[i]

        self.pos += batch_size
        if self.pos >= self.n:
            self.pos = -1

        return x, y, l
