'''
Neuroimaging data classes and utilities
'''

import numpy as np


def resolve(dataset):
    from mri import MRI
    from fmri import FMRI, FMRI_IID
    from snp import SNP

    if dataset == 'mri':
        C = MRI
    elif dataset == 'fmri':
        C = FMRI
    elif dataset == 'fmri_iid':
        C = FMRI_IID
    elif dataset == 'snp':
        C = SNP
    else:
        raise ValueError(dataset)
    return C


def medfilt(x, k):
    '''
    Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    '''
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