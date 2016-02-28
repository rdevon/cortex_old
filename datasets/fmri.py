'''
Module for fMRI data
'''

from collections import OrderedDict
from glob import glob
import numpy as np
import os
from os import path
import pprint
import random
import theano
import yaml

from . import Dataset
import nifti_viewer
import nipy
from nipy.core.api import Image
import rois
from utils import floatX
from utils.tools import resolve_path


def load_data(idx=None, dataset='mri', **dataset_args):
    if dataset == 'mri':
        C = MRI
    else:
        raise ValueErro(dataset)
    train, valid, test, idx = make_datasets(C, **dataset_args)
    return train, valid, test, idx

def make_datasets(C, split=[0.7, 0.2, 0.1], idx=None,
                  train_batch_size=None,
                  valid_batch_size=None,
                  test_batch_size=None,
                  **dataset_args):

    assert C in [MRI]

    if idx is None:
        assert split is not None
        if round(np.sum(split), 5) != 1. or len(split) != 3:
            raise ValueError(split)
        dummy = C(batch_size=1, **dataset_args)
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

    if train_batch_size is not None:
        train = C(idx=idx[0], batch_size=train_batch_size, **dataset_args)
    else:
        train = None
    if valid_batch_size is not None:
        valid = C(idx=idx[1], batch_size=valid_batch_size, **dataset_args)
    else:
        valid = None
    if test_batch_size is not None:
        test = C(idx=idx[2], batch_size=test_batch_size, **dataset_args)
    else:
        test = None

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


class MRI(Dataset):
    def __init__(self, source=None, name='mri', idx=None,
                 distribution='gaussian', **kwargs):
        super(MRI, self).__init__(name=name, **kwargs)

        print 'Loading %s from %s' % (name, source)
        source = resolve_path(source)
        X, Y = self.get_data(source)

        self.dims = {self.name: int(self.mask.sum()),
                     'group': len(np.unique(Y))}
        self.distributions = {self.name: distribution,
                              'group': 'multinomial'}

        self.image_shape = X.shape[1:]
        self.X = self._mask(X)
        self.mean_image = self.X.mean(axis=0)
        self.Y = make_one_hot(Y)

        if distribution == 'gaussian':
            self.X -= self.mean_image
            self.X /= self.X.std()
        elif distribution in ['continuous_binomial', 'binomial']:
            self.X -= self.X.min()
            self.X /= (self.X.max() - self.X.min())
        else:
            raise ValueError(distribution)

        if idx is not None:
            self.X = self.X[idx]
            self.Y = self.Y[idx]

        self.n = self.X.shape[0]

    def _randomize(self):
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.X = self.X[rnd_idx, :]
        self.Y = self.Y[rnd_idx, :]

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
        sites_file = source_dict['sites']

        data_files = source_dict['data']
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

        mask = np.load(mask_file)
        if not np.all(np.bitwise_or(mask == 0, mask == 1)):
            raise ValueError("Mask has incorrect values.")

        self.mask = mask
        self.base_nifti_file = nifti_file
        self.sites = np.load(sites_file).tolist()

        '''
        print 'Regressing out site'
        idx0 = [i for i, s in enumerate(self.sites) if s == 0]
        idx1 = [i for i, s in enumerate(self.sites) if s == 1]
        mi0 = X[idx0].mean(axis=0)
        mi1 = X[idx1].mean(axis=0)

        X[idx0] -= mi0
        X[idx1] -= mi1
        '''
        return X, Y

    def next(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.pos == -1:
            self.reset()
            raise StopIteration

        x = self.X[self.pos:self.pos+batch_size]
        y = self.Y[self.pos:self.pos+batch_size]
        self.pos += batch_size

        if self.pos + batch_size > self.n:
            self.pos = -1

        rval = {
            self.name: x,
            'group': y
        }

        return rval

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

    def save_images(self, x, out_file, remove_niftis=True,
                    order=None, stats=dict(), x_limit=None):
        if len(x.shape) == 3:
            x = x[:, 0, :]
        x = self._unmask(x)

        images, nifti_files = self.save_niftis(x)
        roi_dict = rois.main(nifti_files)

        if remove_niftis:
            for f in nifti_files:
                os.remove(f)
        print 'Saving montage'
        nifti_viewer.montage(images, self.anat_file, roi_dict,
                             out_file=out_file,
                             order=order,
                             stats=stats)
        print 'Done'


class FMRI_IID(MRI):
    def __init__(self, name='fmri_iid', **kwargs):
        super(FMRI_IID, self).__init__(name=name, **kwargs)

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
            print 'Loading %s' % data_file
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
