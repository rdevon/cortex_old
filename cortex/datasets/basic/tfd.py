'''
Module for Toronto faces dataset.
'''

from collections import OrderedDict
import logging
import numpy as np
from os import path
from scipy.io import loadmat

from . import TwoDImageDataset
from ...utils.tools import resolve_path


logger = logging.getLogger(__name__)


class TFD(TwoDImageDataset):
    '''Toronto faces dataset.

    '''
    def __init__(self, source=None, restrict_classes=None, mode='train',
                 fold=0, name='tfd', **kwargs):
        if source is None: raise TypeError('No source file provided')

        logger.info('Loading {name} ({mode}) from {source}'.format(
            name=name, mode=mode, source=resolve_path(source)))

        source = resolve_path(source, fold=fold)

        X, Y, L, image_shape = self.get_data(source, mode)

        if restrict_classes is not None:
            X = np.array(
                [x for i, x in enumerate(X) if Y[i] in restrict_classes])
            Y = np.array(
                [y for i, y in enumerate(Y) if Y[i] in restrict_classes])

        data = {'input': X, 'labels': Y, 'expressions': L}
        distributions = {'input': 'gaussian', 'labels': 'multinomial',
                         'expressions': 'multinomial'}

        super(TFD, self).__init__(data, distributions=distributions,
                                    name=name, mode=mode,
                                    image_shape=image_shape, **kwargs)

    def get_data(self, source, mode, fold=0):
        print 'Loading CIFAR-10 ({mode})'.format(mode=mode)

        d = loadmat(source)
        X = d['images']
        Y_exp = d['lab_exp']
        Y_id = d['lab_id']
        folds = d['folds'][:, fold]

        if mode == 'unlabeled':
            mid = 0
        elif mode == 'train':
            mid = 1
        elif mode == 'valid':
            mid = 2
        elif mode == 'test':
            mid = 3

        idx = [i for i in range(X.shape[0]) if folds[i] == mid]
        X = X[idx]
        Y_exp = Y_exp[idx]
        Y_id = Y_id[idx]

        image_shape = X.shape[1:]
        X = X.reshape((X.shape[0], X.shape[0] * X.shape[1]))

        return X, Y_id, Y_exp, image_shape


_classes = {'CIFAR': CIFAR}