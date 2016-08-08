'''
Module for CIFAR dataset.
'''

from collections import OrderedDict
import cPickle
import gzip
import logging
import numpy as np
from os import path

from . import TwoDImageDataset
from ...utils.tools import resolve_path


logger = logging.getLogger(__name__)


class CIFAR(TwoDImageDataset):
    '''CIFAR dataset.

    '''
    def __init__(self, source=None, restrict_classes=None, mode='train',
                 name='cifar', greyscale=False, **kwargs):
        if source is None: raise TypeError('No source file provided')

        logger.info('Loading {name} ({mode}) from {source}'.format(
            name=name, mode=mode, source=source))

        source = resolve_path(source)

        X, Y = self.get_data(source, mode, greyscale=greyscale)

        if restrict_classes is not None:
            X = np.array(
                [x for i, x in enumerate(X) if Y[i] in restrict_classes])
            Y = np.array(
                [y for i, y in enumerate(Y) if Y[i] in restrict_classes])

        data = {'input': X, 'labels': Y}
        distributions = {'input': 'gaussian', 'labels': 'multinomial'}

        super(CIFAR, self).__init__(data, distributions=distributions,
                                    name=name, mode=mode, image_shape=(32, 32),
                                    greyscale=greyscale, **kwargs)

    def get_data(self, source, mode, greyscale=False):
        print 'Loading CIFAR-10 ({mode})'.format(mode=mode)

        X = []
        Y = []

        if mode == 'train':
            for i in xrange(1, 5):
                with open(path.join(source, 'data_batch_%d' % i)) as f:
                    d = cPickle.load(f)
                    X.append(d['data'])
                    Y.append(d['labels'])
        elif mode == 'valid':
            with open(path.join(source, 'data_batch_5')) as f:
                d = cPickle.load(f)
                X.append(d['data'])
                Y.append(d['labels'])
        elif mode == 'test':
            with open(path.join(source, 'test_batch')) as f:
                d = cPickle.load(f)
                X.append(d['data'])
                Y.append(d['labels'])
        else:
            raise ValueError()

        X = np.concatenate(X)
        Y = np.concatenate(Y)

        if greyscale:
            div = X.shape[1] // 3
            X_r = X[:, :div]
            X_b = X[:, div:2*div]
            X_g = X[:, 2*div:]
            X = (X_r + X_b + X_g) / 3.0

        X = X.astype('float32') / float(255)

        return X, Y


_classes = {'CIFAR': CIFAR}