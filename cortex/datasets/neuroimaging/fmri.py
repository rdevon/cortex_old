'''
Module for fMRI data
'''

import cPickle
from collections import OrderedDict
from glob import glob
import nipy
from nipy.core.api import Image
import numpy as np
import os
from os import path
import pprint
import random
import theano
import yaml

from ...analysis.mri import rois
from .. import Dataset, make_one_hot
from .mri import MRI
from . import nifti_viewer
from ...utils import floatX
from ...utils.tools import resolve_path


class FMRI_IID(MRI):
    '''fMRI data treated as IID.

    Use this dataset if you plan to use a model that needs identical and
    independently sampled data.

    Attributes:
        novels

    '''
    def __init__(self, name='fmri_iid', **kwargs):
        super(FMRI_IID, self).__init__(name=name, **kwargs)

    def get_data(self, source):
        print('Loading file locations from %s' % source)
        source_dict = yaml.load(open(source))
        print('Source locations: %s' % pprint.pformat(source_dict))

        def unpack_source(nifti_file=None, mask_file=None, anat_file=None,
                          tmp_path=None, pca=None, data=None, **kwargs):
            return (nifti_file, mask_file, anat_file, tmp_path, pca, data, kwargs)

        (nifti_file, mask_file, self.anat_file,
         self.tmp_path, pca_file, data_files, extras) = unpack_source(
            **source_dict)

        self.base_nifti_file = nifti_file
        if not path.isdir(self.tmp_path):
            os.mkdir(self.tmp_path)

        mask = np.load(mask_file)
        if not np.all(np.bitwise_or(mask == 0, mask == 1)):
            raise ValueError("Mask has incorrect values.")
        self.mask = mask

        if pca_file is not None:
            with open(pca_file, 'rb') as f:
                self.pca = cPickle.load(f)
        else:
            self.pca = None

        if isinstance(data_files, str):
            data_files = [data_files]
        X = []
        Y = []
        for i, data_file in enumerate(data_files):
            print 'Loading %s' % data_file
            X_ = np.load(data_file)
            X.append(X_.astype(floatX))
            Y.append((np.zeros((X_.shape[0],)) + i).astype(floatX))

        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)

        self.targets = np.load(targets_file)
        self.novels = np.load(novels_file)

        self.n_scans = self.targets.shape[0]
        self.n_subjects = X.shape[0] // self.n_scans

        return X, Y


class FMRI(FMRI_IID):
    '''fMRI dataset class.

    Treats fMRI as sequences, instead as IID as with FMRI_IID.

    Attributes:
        window (int): window size of fMRI batches.
        stride (int): stride of fMRI batches.
        n (int): number of subjects.
        idx (list): indices of subject, scan-window pairs.

    '''

    def __init__(self, name='fmri', window=10, stride=1, idx=None, **kwargs):
        '''Init function for fMRI.

        Args:
            name (str): name of dataset.
            window (int): window size of fMRI batches.
            stride (int): stride of fMRI batches.
            idx (list): indices of dataset (subjects).
            **kwargs: keyword arguments for initializaiton.

        '''
        super(FMRI, self).__init__(name=name, **kwargs)

        self.window = window
        self.stride = stride

        self.X = self.X.reshape((self.n_subjects, self.n_scans, self.X.shape[1]))
        self.Y = self.Y.reshape((self.n_subjects, self.n_scans, self.Y.shape[1]))

        if idx is not None:
            self.X = self.X[idx]
            self.Y = self.Y[idx]
            self.n_subjects = len(idx)

        scan_idx = range(0, self.n_scans - window + 1, stride)
        scan_idx_e = scan_idx * self.n_subjects
        subject_idx = range(self.n_subjects)
        # Similar to np.repeat, but using list comprehension.
        subject_idx_e = [i for j in [[s] * len(scan_idx) for s in subject_idx]
                         for i in j]
        # idx is list of (subject, scan)
        self.idx = zip(subject_idx_e, scan_idx_e)
        self.n = len(self.idx)

        if self.shuffle:
            self.randomize()

    def randomize(self):
        '''Randomize the fMRI dataset.

        Shuffles the idx.

        '''
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.idx = [self.idx[i] for i in rnd_idx]

    def next(self, batch_size=None):
        '''Draws the next batch of windowed fMRI.

        Args:
            batch_size (Optional[int]): number of windows in batch.

        Returns:
            dict: dictionary of batched data

        '''
        if batch_size is None:
            batch_size = self.batch_size

        if self.pos == -1:
            self.reset()
            raise StopIteration

        idxs = [self.idx[i] for i in range(self.pos, self.pos+batch_size)]
        x = np.array([self.X[i][j:j+self.window] for i, j in idxs]).astype(floatX).transpose(1, 0, 2)
        y = np.array([self.Y[i][j:j+self.window] for i, j in idxs]).astype(floatX).transpose(1, 0, 2)

        self.pos += batch_size

        if self.pos + batch_size > self.n:
            self.pos = -1

        rval = {
            self.name: x,
            'group': y
        }

        return rval


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
