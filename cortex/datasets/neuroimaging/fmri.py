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
from . import mri as mri_module
from ...analysis import nifti_viewer
from ...utils import floatX
from ...utils.tools import resolve_path


class FMRI_IID(mri_module.MRI):
    '''fMRI data treated as IID.

    Use this dataset if you plan to use a model that needs identical and
    independently sampled data.

    Attributes:
        extras (dict): dictionary of additional arrays for analysis.
        clean_data (bool): remove subjects with 0-std voxels after masking.

    '''
    def __init__(self, name='fmri_iid', detrend=False, load_preprocessed=True,
                 **kwargs):
        self.detrend = detrend
        self.load_preprocessed = load_preprocessed
        super(FMRI_IID, self).__init__(name=name, **kwargs)

    def get_data(self, source):
        '''Fetch the fMRI dataset.

        fMRI dataset source is a yaml file.
        An example format of said yaml is::

            name: 'aod',
            data: [
                '/Users/devon/Data/AOD/AOD_0.npy',
                '/Users/devon/Data/AOD/AOD_1.npy'
            ],
            mask: '/Users/devon/Data/AOD/AOD_mask.npy',
            nifti: '//Users/devon/Data/VBM/H000A.nii',
            tmp_path: '/Users/devon/Data/mri_tmp/',
            anat_file: '/Users/devon/Data/ch2better_whitebg_aligned2EPI_V4.nii',

        '''
        self.logger.info('Loading file locations from %s' % source)
        source_dict = yaml.load(open(source))
        self.logger.debug('Source locations: \n%s' % pprint.pformat(source_dict))

        def unpack_source(name=None, nifti=None, mask=None, anat_file=None,
                          tmp_path=None, pca=None, data=None, preprocessed=None,
                          **kwargs):
            return (name, nifti, mask, anat_file, tmp_path, pca, data,
                    preprocessed, kwargs)

        (name, nifti_file, mask_file, anat_file,
         tmp_path, pca_file, data_files, preprocessed, extras) = unpack_source(
            **source_dict)
        nifti_file = resolve_path(nifti_file)
        mask_file = resolve_path(mask_file)
        self.anat_file = resolve_path(anat_file)
        self.tmp_path = resolve_path(tmp_path)
        self.pca_file = resolve_path(pca_file)

        self.update_progress()

        self.base_nifti_file = nifti_file
        if not path.isdir(self.tmp_path): os.mkdir(self.tmp_path)

        mask = np.load(mask_file)
        if not np.all(np.bitwise_or(mask == 0, mask == 1)):
            raise ValueError("Mask has incorrect values.")
        self.mask = mask
        self.update_progress()

        if self.pca_file is not None:
            try:
                with open(self.pca_file, 'rb') as f:
                    self.pca = cPickle.load(f)
            except (IOError, EOFError):
                self.pca = None
        else:
            self.pca = None
        self.update_progress()

        self.extras = dict((k, np.load(resolve_path(v)).astype(floatX))
            for k, v in extras.iteritems())

        if preprocessed is not None: preprocessed = resolve_path(preprocessed)

        if (not self.load_preprocessed
            or preprocessed is None
            or not path.isfile(preprocessed)):
            if isinstance(data_files, str):
                data_files = [data_files]
            X = []
            Y = []
            for i, data_file in enumerate(data_files):
                data_file = resolve_path(data_file)
                self.logger.info('Loading %s' % data_file)
                self.update_progress(progress=False)
                X_ = np.load(data_file)
                X.append(X_.astype(floatX))
                Y.append((np.zeros((X_.shape[0], X_.shape[1])) + i).astype(floatX))
            self.update_progress()

            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)

            if len(X.shape) == 3:
                self.n_subjects, self.n_scans, _ = X.shape
            elif len(X.shape) == 5:
                self.n_subjects, self.n_scans, _, _, _ = X.shape
            else:
                raise ValueError('X has incorrect shape. Should be 3 or 5 (got %d)'
                                 % len(X.shape))
            X = self._mask(X)

            if self.detrend:
                self.logger.info('Detrending voxels...')
                X = self.perform_detrend(X.transpose(1, 0, 2)).transpose(1, 0, 2)

            X -= X.mean(axis=1, keepdims=True)
            X /= np.sqrt(X.std(axis=1, keepdims=True) ** 2 + 1e-6)
            X = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))
            X -= X.mean(axis=0, keepdims=True)
            X /= X.std(axis=0, keepdims=True)
            Y = Y.reshape((Y.shape[0] * Y.shape[1]))
        else:
            self.logger.info('Reloading preprocessed from {}'.format(preprocessed))
            self.update_progress()
            d = np.load(preprocessed)
            X = d['X']
            Y = d['Y']
            self.n_subjects = int(d['n_subjects'])
            self.n_scans = int(d['n_scans'])

        if (self.load_preprocessed
            and preprocessed is not None
            and not path.isfile(preprocessed)):
            d = dict(X=X, Y=Y, n_subjects=self.n_subjects, n_scans=self.n_scans)
            self.logger.info('Saving preprocessed to {}'.format(preprocessed))
            np.savez(preprocessed, **d)

        self.update_progress()

        return X.astype(floatX), Y

    def perform_detrend(self, data, order=4):
        x = np.arange(data.shape[0])
        if len(data.shape) == 3:
            shape = data.shape
            data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
        elif len(data.shape) > 3:
            raise ValueError('Detrending over 3 dims not supported')
        else:
            reshape = None
        fit = np.polyval(np.polyfit(x, data, deg=order),
                         np.repeat(x[:, None], data.shape[1], axis=1))
        data = data - fit
        if shape is not None:
            data = data.reshape(shape)
        return data.astype(floatX)

class FMRI(FMRI_IID):
    '''fMRI dataset class.

    Treats fMRI as sequences, instead as IID as with FMRI_IID.

    Attributes:
        window (int): window size of fMRI batches.
        stride (int): stride of fMRI batches.
        n (int): number of subjects.
        idx (list): indices of subject, scan-window pairs.

    '''

    def __init__(self, name='fmri', window=10, stride=1, **kwargs):
        '''Init function for fMRI.

        Args:
            name (str): name of dataset.
            window (int): window size of fMRI batches.
            stride (int): stride of fMRI batches.
            idx (list): indices of dataset (subjects).
            **kwargs: keyword arguments for initializaiton.

        '''
        self.idx = None
        self.window = window
        self.stride = stride

        super(FMRI, self).__init__(name=name, **kwargs)

    def finish_setup(self):
        self.X = self.X.reshape(
            (self.n_subjects, self.n_scans, self.X.shape[1]))
        self.Y = self.Y.reshape(
            (self.n_subjects, self.n_scans, self.Y.shape[1]))
        for k, v in self.data.iteritems():
            self.data[k] = v.reshape((self.n_subjects, self.n_scans, v.shape[1]))
        self.set_idx()

    def set_idx(self):
        scan_idx = range(0, self.n_scans - self.window + 1, self.stride)
        scan_idx_e = scan_idx * self.n_subjects
        subject_idx = range(self.n_subjects)
        subject_idx_e = [i for j in [[s] * len(scan_idx) for s in subject_idx]
                         for i in j]
        self.idx = zip(subject_idx_e, scan_idx_e)
        self.n_samples = len(self.idx)

    def slice_data(self, idx):
        super(FMRI, self).slice_data(idx)
        self.set_idx()
        if self.shuffle: self.randomize()

    def randomize(self):
        '''Randomize the fMRI dataset.

        Shuffles the idx.

        '''
        if self.idx is None: self.set_idx()
        rnd_idx = np.random.permutation(np.arange(0, len(self.idx), 1))
        self.idx = [self.idx[i] for i in rnd_idx]

    def next(self, batch_size):
        '''Draws the next batch of windowed fMRI.

        Args:
            batch_size (int): number of windows in batch.

        Returns:
            dict: dictionary of batched data

        '''

        if self.pos == -1:
            self.reset()
            raise StopIteration

        idxs = [self.idx[i] for i in range(self.pos, self.pos+batch_size)]
        x = np.array(
            [self.X[i][j:j+self.window] for i, j in idxs]).transpose(1, 0, 2)
        y = np.array(
            [self.Y[i][j:j+self.window] for i, j in idxs]).transpose(1, 0, 2)
        y_l = y[0]
        x_c = self.X.transpose(1, 0, 2)
        targets = np.array(
            [self.extras['targets'][j:j+self.window] for i, j in idxs])
        novels = np.array(
            [self.extras['novels'][j:j+self.window] for i, j in idxs])
        stim = np.concatenate([targets[:, None], novels[:, None]], axis=1).astype(floatX)
        stim_all = np.concatenate([self.extras['targets'][:, None],
                                   self.extras['novels'][:, None]], axis=1)

        self.pos += batch_size
        if self.pos + batch_size > self.n_samples: self.pos = -1

        return dict(input=x, labels=y, labels_t=y_l, x_total=x_c, stim=stim,
                    stim_all=stim_all)

    def set_mean(self, x):
        shape = x.shape
        x = x.reshape((shape[0] * shape[1], shape[2]))
        x = self.prepare_images(x)
        x = x.reshape((shape[0], shape[1],) + tuple(x.shape[1:]))
        self.temporal_mean = x.mean(axis=0)

    def viz(self, x, time_course_keys=None, t_limit=None, **kwargs):
        if time_course_keys is not None:
            time_courses = dict()
            for k in time_course_keys:
                time_courses[k] = kwargs.pop(k)
        else:
            time_courses = None

        if time_courses is not None:
            targets = (self.extras['targets'] - self.extras['targets'].mean()) / self.extras['targets'].std()
            novels = (self.extras['novels'] - self.extras['novels'].mean()) / self.extras['novels'].std()

            if isinstance(time_courses, np.ndarray):
                time_courses = {'tc': time_courses}
            elif isinstance(time_courses, list):
                time_courses = dict(('tc%d' % i, tc)
                    for i, tc in enumerate(time_courses))
            elif not isinstance(time_courses, dict):
                raise TypeError('Time courses must be dict, list, or np.ndarray.')

            time_courses['targets'] = targets
            time_courses['novels'] = novels

            for k in time_courses.keys():
                tc = time_courses[k]
                t_limit = t_limit or tc.shape[0]
                if tc.ndim == 3: tc = tc.mean(1)

                tc = tc - tc.mean(axis=0, keepdims=True)
                tc = tc / tc.std(axis=0, keepdims=True)
                tc = tc[:t_limit]

                if tc.ndim == 2: tc = tc.T

                time_courses[k] = tc

        super(FMRI, self).viz(x, time_courses=time_courses, **kwargs)

    def viz_mean(self, x, out_file=None, remove_niftis=True, roi_dict=None,
            stats=None, update_rois=True, global_norm=False, **kwargs):
        shape = x.shape
        if roi_dict is None: roi_dict = dict()
        x_tot = []
        nC = x.shape[1]
        for c in xrange(nC):
            x_c = x[:, c]
            x_c = self.prepare_images(x_c)
            x_tot.append()
        x = x_tot
        x /= nT
        x = self._unmask(x)
        if global_norm:
            global_std = self.global_norm
        else:
            global_std = None
        images, nifti_files = self.save_niftis(x)

        if update_rois: roi_dict.update(**rois.main(nifti_files))
        if stats is None: stats = dict()
        stats['gm'] = [v['top_clust']['grey_value'] for v in roi_dict.values()]

        if remove_niftis:
            for f in nifti_files:
                os.remove(f)
        nifti_viewer.montage(images, self.anat_file, roi_dict,
                             out_file=resolve_path(out_file), stats=stats,
                             global_std=global_std, **kwargs)

    def viz_std(self, x, out_file=None, remove_niftis=True, roi_dict=None,
            stats=None, update_rois=True, global_norm=False, **kwargs):
        x = x[:, 0]
        shape = x.shape
        if roi_dict is None: roi_dict = dict()
        x_tot = []
        nC = x.shape[1]
        for c in xrange(nC):
            x_c = x[:, c]
            x_c = self.prepare_images(x_c)
            x_tot.append(x_c.std(0))
        x = np.array(x_tot)
        x = self._unmask(x)
        if global_norm:
            global_std = self.global_norm
        else:
            global_std = None
        images, nifti_files = self.save_niftis(x)

        if update_rois: roi_dict.update(**rois.main(nifti_files))
        if stats is None: stats = dict()
        stats['gm'] = [v['top_clust']['grey_value'] for v in roi_dict.values()]

        if remove_niftis:
            for f in nifti_files:
                os.remove(f)
        nifti_viewer.montage(images, self.anat_file, roi_dict,
                             out_file=resolve_path(out_file), stats=stats,
                             global_std=global_std, **kwargs)

    def viz_unfold(self, x, out_file=None, remove_niftis=True, **kwargs):
        if len(x.shape) == 3:
            shape = x.shape
            x = x.reshape((shape[0] * shape[1], shape[2]))
        else:
            shape = None
        x = self.prepare_images(x)
        if shape is not None:
            x = x.reshape((shape[0], shape[1], x.shape[1]))
            x = x.mean(1) / x.std(1)
        image_std = x.std()
        image_max = x.max()

        if len(x.shape) == 3: x = x[:, 0, :]
        x = self._unmask(x)
        images, nifti_files = self.save_niftis(x)

        nifti_viewer.unfolded_movie(
            images, nifti_files, self.anat_file, out_file=out_file,
            image_max=image_max, image_std=image_std,
            stimulus=dict(targets=self.extras['targets'],
                          novels=self.extras['novels']),
            **kwargs)
        if remove_niftis:
            for f in nifti_files:
                os.remove(f)

_classes = {'FMRI_IID': FMRI_IID, 'FMRI': FMRI}
