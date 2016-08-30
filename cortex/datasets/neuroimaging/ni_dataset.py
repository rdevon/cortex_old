'''Factory for Neuroimaging datasets with subjects.

'''

import logging
import numpy as np
import os
from os import path
import random
import yaml

from .. import BasicDataset
from ...utils.tools import resolve_path
from ...utils.extra import download_data, unzip


def fetch_neuroimaging_data():
    '''Fetch the neuroimaging dataset for demos.

    '''
    url = 'http://mialab.mrn.org/data/neuroimaging/neuroimaging.zip'
    out_dir = resolve_path('$data')
    download_data(url, out_dir)
    unzip(path.join(out_dir, 'neuroimaging.zip'), out_dir)
    os.remove(path.join(out_dir, 'neuroimaging.zip'))

    ni_dir = path.join(out_dir, 'neuroimaging')

    unzip(path.join(ni_dir, 'VBM_test.zip'), ni_dir)
    os.remove(path.join(ni_dir, 'VBM_test.zip'))

    unzip(path.join(ni_dir, 'AOD_test.zip'), ni_dir)
    os.remove(path.join(ni_dir, 'AOD_test.zip'))

    yaml_file = path.join(ni_dir, 'VBM_test', 'VBM.yaml')
    with open(yaml_file, 'w') as yf:
        yf.write(
            yaml.dump(
                dict(
                    anat_file=path.join(ni_dir, 'ch2better_whitebg_aligned2EPI_V4.nii'),
                    data=[path.join(ni_dir, 'VBM_test', 'VBM_0.npy'),
                          path.join(ni_dir, 'VBM_test', 'VBM_1.npy')],
                    mask=path.join(ni_dir, 'VBM_test', 'VBM_mask.npy'),
                    name='VBM',
                    nifti=path.join(ni_dir, 'base_nifti.nii'),
                    sites=path.join(ni_dir, 'VBM_test', 'VBM_sites.npy'),
                    tmp_path=path.join(ni_dir, 'VBM_test', 'VBM_tmp')
                    )
                )
            )

    yaml_file = path.join(ni_dir, 'AOD_test', 'AOD.yaml')
    with open(yaml_file, 'w') as yf:
        yf.write(
            yaml.dump(
                dict(
                    anat_file=path.join(ni_dir, 'ch2better_whitebg_aligned2EPI_V4.nii'),
                    data=[path.join(ni_dir, 'AOD_test', 'AOD_0.npy'),
                          path.join(ni_dir, 'AOD_test', 'AOD_1.npy')],
                    mask=path.join(ni_dir, 'AOD_test', 'AOD_mask.npy'),
                    name='AOD',
                    nifti=path.join(ni_dir, 'base_nifti.nii'),
                    tmp_path=path.join(ni_dir, 'AOD_test', 'AOD_tmp')
                    )
                )
            )

def medfilt(x, k):
    '''
    Apply a length-k median filter to a 1D array x.

    Boundaries are extended by repeating endpoints.

    Args:
        x (numpy.array)
        k (int)

    Returns:
        numpy.array
    '''
    assert k % 2 == 1, 'Median filter length must be odd.'
    assert x.ndim == 1, 'Input must be one-dimensional.'
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


class NeuroimagingDataset(BasicDataset):
    def slice_data(self, idx, axis=0):
        for k, v in self.data.iteritems():
            if axis == 0:
                self.data[k] = v[idx]
            elif axis == 1:
                self.data[k] = v[:, idx]
            else:
                raise TypeError
        self.X = self.data['input']
        if self.labels in self.data.keys():
            self.Y = self.data[self.labels]
        self.n_subjects = len(idx)
        self.n_samples = self.X.shape[0]

    def update_progress(self, finish=False, progress=True):
        if progress:
            self.progress += 1
        if finish:
            self.pbar.update(self._init_steps)
        else:
            self.pbar.update(self.progress)

    @classmethod
    def factory(C, split=None, idx=None, **kwargs):
        from ... import _manager as manager

        if split is None and idx is None:
            raise TypeError('`split` or `idx` must be provided')

        data = C(batch_size=10, mode='dummy', **kwargs)
        if hasattr(data, 'logger'):
            logger = data.logger
            data.logger = None
        else:
            logger = None

        if idx is None:
            logger.info('Splitting dataset into ratios %r' % split)
            if round(np.sum(split), 5) != 1. or len(split) != 3:
                raise TypeError('`split` must be of length 3 and sum to 1.')

            if data.balance:
                l_idx = [np.where(label == 1)[0].tolist() for label in data.Y[:, 0, :].T]
                train_idx = []
                valid_idx = []
                test_idx = []
                for l in l_idx:
                    random.shuffle(l)
                    split_idx = []
                    accum = 0
                    for s in split:
                        s_i = int(s * len(l) + accum)
                        split_idx.append(s_i)
                        accum += s_i
                    train_idx += l[:split_idx[0]]
                    valid_idx += l[split_idx[0]:split_idx[1]]
                    test_idx += l[split_idx[1]:]
            else:
                split_idx = []
                accum = 0
                for s in split:
                    s_i = int(s * data.n_subjects + accum)
                    split_idx.append(s_i)
                    accum += s_i
                idx = range(data.n_subjects)
                random.shuffle(idx)

                train_idx = idx[:split_idx[0]]
                valid_idx = idx[split_idx[0]:split_idx[1]]
                test_idx = idx[split_idx[1]:]
            idx = [train_idx, valid_idx, test_idx]
        else:
            if len(idx) != 3:
                raise TypeError('`idx` must be length 3')
            logger.info('Splitting dataset into ratios %.2f / %.2f / %.2f '
                        'using given indices'
                        % tuple(len(idx[i]) / float(data.n_subjects)
                                for i in range(3)))

        modes = ['train', 'valid', 'test']
        for i, mode in zip(idx, modes):
            dataset = data.copy()
            dataset.slice_data(i)
            dataset.logger = logger
            dataset.mode = mode
            dataset.register()
            logger.debug('%s dataset has %d subjects'
                         % (dataset.mode, len(i)))

        manager.datasets[data.name]['idx'] = idx