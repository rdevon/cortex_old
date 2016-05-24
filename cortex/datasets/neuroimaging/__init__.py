'''
Neuroimaging data classes and utilities
'''

import numpy as np
import os
from os import path
import yaml

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
                    anat_file=path.join(ni_dir, 'ch2better_whitebg_aligned2EPI_V4'),
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
                    anat_file=path.join(ni_dir, 'ch2better_whitebg_aligned2EPI_V4'),
                    data=[path.join(ni_dir, 'AOD_test', 'AOD_0.npy'),
                          path.join(ni_dir, 'AOD_test', 'AOD_1.npy')],
                    mask=path.join(ni_dir, 'AOD_test', 'AOD_mask.npy'),
                    name='AOD',
                    nifti=path.join(ni_dir, 'base_nifti.nii'),
                    tmp_path=path.join(ni_dir, 'AOD_test', 'AOD_tmp')
                    )
                )
            )

def resolve(dataset):
    '''Resolve neuroimaging dataset.

    Args:
        dataset (str): dataset name

    '''
    from .fmri import FMRI, FMRI_IID
    from .mri import MRI
    from .snp import SNP

    if dataset == 'fmri':
        C = FMRI
    elif dataset == 'fmri_iid':
        C = FMRI_IID
    elif dataset == 'mri':
        C = MRI
    elif dataset == 'snp':
        C = SNP
    else:
        raise ValueError(dataset)
    return C


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