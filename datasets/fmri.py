'''
Module for fMRI data
'''

from glob import glob
import numpy as np
import os
from os import path
import pprint
import random
import theano
import yaml

import nifti_viewer
import nipy
from nipy.core.api import Image
import rois


floatX = theano.config.floatX

def load_data(idx=None, **dataset_args):
    train, valid, test, idx = make_datasets(MRI, idx=idx, **dataset_args)
    return train, valid, test, idx

def make_datasets(C, split=None, idx=None, **dataset_args):
    assert C in [MRI]

    if idx is None:
        assert split is not None
        if round(np.sum(split), 5) != 1. or len(split) != 3:
            raise ValueError(split)
        dummy = C(**dataset_args)
        N = dummy.n
        idx = range(N)
        random.shuffle(idx)
        split_idx = []
        accum = 0
        for s in split:
            s_i = int(s * N + accum)
            split_idx.append(s_i)
            accum += s_i

        train_idx = idx[:split_idx[0]]
        valid_idx = idx[split_idx[0]:split_idx[1]]
        test_idx = idx[split_idx[1]:]
        idx = [train_idx, valid_idx, test_idx]

    train = C(idx=idx[0], **dataset_args)
    valid = C(idx=idx[1], **dataset_args)
    test = C(idx=idx[2], **dataset_args)

    return train, valid, test, idx

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


class MRI(object):
    def __init__(self, source=None, batch_size=None,
                 shuffle=True, idx=None):
        print 'Loading MRI from %s' % source

        X, Y = self.get_data(source)

        self.image_shape = X.shape[1:]
        self.dims = dict(mri=int(self.mask.sum()), group=len(np.unique(Y)))
        self.acts = dict(mri='lambda x: x', group='T.nnet.softmax')
        self.shuffle = shuffle
        self.pos = 0
        self.batch_size = batch_size
        self.next = self._next

        if idx is not None:
            X = X[idx]
            Y = Y[idx]

        self.n = X.shape[0]

        self.X = self._mask(X)
        self.Y = make_one_hot(Y)

        self.mean_image = self.X.mean(axis=0)

        if self.shuffle:
            self._randomize()

    def _randomize(self):
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.X = self.X[rnd_idx, :]
        self.Y = self.Y[rnd_idx, :]

    def _reset(self):
        self.pos = 0
        if self.shuffle:
            self._randomize()

    def _next(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.pos == -1:
            self._reset()
            raise StopIteration

        x = self.X[self.pos:self.pos+batch_size]
        y = self.Y[self.pos:self.pos+batch_size]
        self.pos += batch_size

        if self.pos + batch_size > self.n:
            self.pos = -1

        return x, y

    def get_data(self, source):
        print('Loading file locations from %s' % source)
        source_dict = yaml.load(open(source))
        print('Source locations: %s' % pprint.pformat(source_dict))

        nifti_file = source_dict['nifti']
        mask_file = source_dict['mask']
        self.tmp_path = source_dict['tmp_path']
        if not path.isdir(self.tmp_path):
            os.mkdir(self.tmp_path)
        self.anat_file = source_dict['anat_file']

        data_files = source_dict['data']
        if isinstance(data_files, str):
            data_files = [data_files]

        X = []
        Y = []
        for i, data_file in enumerate(data_files):
            X_ = np.load(data_file)
            X.append(X_.astype(floatX))
            Y.append((np.zeros((X_.shape[0],)) + i).astype(floatX))

        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)

        mask = np.load(mask_file)
        if not np.all(np.bitwise_or(mask == 0, mask == 1)):
            raise ValueError("Mask has incorrect values.")

        self.mask = mask
        self.base_nifti_file = nifti_file

        return X, Y

    def _mask(self, X, mask=None):
        if mask is None:
            mask = self.mask

        if X.shape[1:] != mask.shape:
            raise ValueError()

        mask_f = mask.flatten()
        mask_idx = np.where(mask_f == 1)[0].tolist()
        X_masked = np.zeros((X.shape[0], self.dims['mri'])).astype(floatX)

        for i, x in enumerate(X):
            X_masked[i] = x.flatten()[mask_idx]

        return X_masked

    def _unmask(self, X_masked, mask=None):
        if mask is None:
            mask = self.mask

        if X_masked.shape[1] != self.dims['mri']:
            raise ValueError(X_masked.shape)

        mask_f = mask.flatten()
        mask_idx = np.where(mask_f == 1)[0].tolist()
        X = np.zeros((X_masked.shape[0],) + self.image_shape).astype(floatX)

        for i, x_m in enumerate(X_masked):
            x_f = X[i].flatten()
            x_f[mask_idx] = x_m
            X[i] = x_f.reshape(self.image_shape)

        return X

    def make_image(self, X, base_nifti):
        image = Image.from_image(base_nifti, data=X)
        return image

    def save_niftis(self, X):
        base_nifti = nipy.load_image(self.base_nifti_file)

        images = []
        out_files = []
        for i, x in enumerate(X):
            image = self.make_image(x, base_nifti)
            out_file = path.join(self.tmp_path, 'tmp_image_%d.nii.gz' % i)
            nipy.save_image(image, out_file)
            images.append(image)
            out_files.append(out_file)

        return images, out_files

    def save_images(self, x, out_file, remove_niftis=True):
        x = self._unmask(x)

        images, nifti_files = self.save_niftis(x)
        roi_dict = rois.main(nifti_files)

        if remove_niftis:
            for f in nifti_files:
                os.remove(f)

        print 'Saving montage'
        nifti_viewer.montage(images, self.anat_file, roi_dict, out_file=out_file)
        print 'Done'


class FMRI_IID(object):
    def __init__(self, source):
        print 'Loading '
        pass

    def randomize(self):
        pass

    def __iter__(self):
        return self

    def next(self, batch_size):
        pass

    def name(self, ):
        pass

    def save_images(self, x, outfile):
        pass

    def show(self, image):
        pass


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
