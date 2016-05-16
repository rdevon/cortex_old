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
from sklearn.decomposition import PCA
import theano
import yaml

from analysis.mri import rois
from . import Dataset
import nifti_viewer
from utils import floatX
from utils.tools import resolve_path


class MRI(Dataset):
    def __init__(self, source=None, name='mri', idx=None,
                 pca_components=0, distribution='gaussian', **kwargs):
        super(MRI, self).__init__(name=name, **kwargs)

        print 'Loading %s from %s' % (name, source)
        source = resolve_path(source)
        X, Y = self.get_data(source)

        self.image_shape = self.mask.shape
        self.X = self._mask(X)
        self.Y = make_one_hot(Y)
        self.pca_components = pca_components

        if self.pca_components and self.pca is None:
            self.pca = PCA(pca_components)
            print 'Performing PCA...'
            self.X = self.pca.fit_transform(self.X)

        self.dims = {self.name: self.X.shape[1],
                     'group': len(np.unique(Y))}
        self.distributions = {self.name: distribution,
                              'group': 'multinomial'}

        if distribution == 'gaussian':
            self.X -= self.X.mean(axis=0)
            self.X /= self.X.std()
        elif distribution in ['continuous_binomial', 'binomial']:
            self.X -= self.X.min()
            self.X /= (self.X.max() - self.X.min())
        else:
            raise ValueError(distribution)

        self.mean_image = self.X.mean(axis=0)

        if idx is not None:
            self.X = self.X[idx]
            self.Y = self.Y[idx]

        self.n = self.X.shape[0]

    def randomize(self):
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
        pca_file = source_dict.get('pca', None)
        if pca_file is not None:
            with open(pca_file, 'rb') as f:
                self.pca = cPickle.load(f)
        else:
            self.pca = None

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

        if 'sites' in source_dict.keys():
            sites_file = source_dict['sites']
            self.sites = np.load(sites_file).tolist()
            print 'Regressing out site'
            idx0 = [i for i, s in enumerate(self.sites) if s == 0]
            idx1 = [i for i, s in enumerate(self.sites) if s == 1]
            mi0 = X[idx0].mean(axis=0)
            mi1 = X[idx1].mean(axis=0)

            X[idx0] -= mi0
            X[idx1] -= mi1

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

        if X.shape[1] == mask.sum():
            print 'Data already masked'
            return X

        if X.shape[1:] != mask.shape:
            raise ValueError((X.shape, mask.shape))

        mask_f = mask.flatten()
        mask_idx = np.where(mask_f == 1)[0].tolist()
        X_masked = np.zeros((X.shape[0], mask.sum())).astype(floatX)

        for i, x in enumerate(X):
            X_masked[i] = x.flatten()[mask_idx]

        return X_masked

    def _unmask(self, X_masked, mask=None):
        if mask is None:
            mask = self.mask

        if X_masked.shape[1] != mask.sum():
            raise ValueError('Masked data does not fit mask %r vs %r' % (X_masked.shape, mask.sum()))

        mask_f = mask.flatten()
        mask_idx = np.where(mask_f == 1)[0].tolist()
        X = np.zeros((X_masked.shape[0],) + self.image_shape).astype(floatX)
        for i, x_m in enumerate(X_masked):
            x_f = X[i].flatten()
            x_f[mask_idx] = x_m
            X[i] = x_f.reshape(self.image_shape)

        return X

    def make_image(self, X, base_nifti, do_pca=True):
        if self.pca is not None and do_pca and self.pca_components:
            X = self.pca.inverse_transform(X)
        image = Image.from_image(base_nifti, data=X)
        return image

    def save_niftis(self, X):
        base_nifti = nipy.load_image(self.base_nifti_file)

        if self.pca is not None and self.pca_components:
            X = self.pca.inverse_transform(X)

        images = []
        out_files = []
        for i, x in enumerate(X):
            image = self.make_image(x, base_nifti, do_pca=False)
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
        nifti_viewer.montage(images, self.anat_file, roi_dict,
                             out_file=out_file,
                             order=order,
                             stats=stats)


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
        pca_file = source_dict.get('pca', None)
        if pca_file is not None:
            with open(pca_file, 'rb') as f:
                self.pca = cPickle.load(f)
        else:
            self.pca = None

        data_files = source_dict['data']
        if isinstance(data_files, str):
            data_files = [data_files]

        targets_file = source_dict['targets']
        novels_file = source_dict['novels']

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
        self.targets = np.load(targets_file)
        self.novels = np.load(novels_file)

        self.n_scans = self.targets.shape[0]
        self.n_subjects = X.shape[0] // self.n_scans

        return X, Y


class FMRI(FMRI_IID):
    '''fMRI dataset class.

    Treats fMRI as sequences, instead as IID as with FMRI_IID.
    '''
    def __init__(self, name='fmri', window=10, stride=1, idx=None, **kwargs):
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
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.idx = [self.idx[i] for i in rnd_idx]

    def next(self, batch_size=None):
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
