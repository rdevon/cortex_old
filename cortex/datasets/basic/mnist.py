'''MNIST dataset.

'''

from collections import OrderedDict
import cPickle
import gzip
import logging
import numpy as np
import random

from . import TwoDImageDataset
from ...utils import concatenate, scan, _rng
from ...utils.tools import resolve_path


logger = logging.getLogger(__name__)


class MNIST(TwoDImageDataset):
    '''MNIST dataset iterator.

    Attributes:
        image_shape (tuple): dimensions of original images.

    '''
    _viz = ['classification_visualization']

    def __init__(self, source=None, restrict_digits=None, mode='train',
                 binarize=False, name='mnist', **kwargs):
        '''Init function for MNIST.

        Args:
            source (str): Path to source gzip file.
            restrict_digits (Optional[list]): list of digits to restrict
                iterator to.
            mode (str): `train`, `test`, or `valid`.
            out_path (Optional[str]): path for saving visualization output.
            name (str): name of dataset.
            **kwargs: eXtra keyword arguments passed to BasicDataset

        '''
        if source is None:
            raise TypeError('No source file provided')
        
        self.source = source

        logger.info('Loading {name} ({mode}) from {source}'.format(
            name=name, mode=mode, source=source))

        source = resolve_path(source)

        X, Y = self.get_data(source, mode)

        if restrict_digits is not None:
            X = np.array(
                [x for i, x in enumerate(X) if Y[i] in restrict_digits])
            Y = np.array(
                [y for i, y in enumerate(Y) if Y[i] in restrict_digits])

        data = {'input': X, 'labels': Y}
        distributions = {'input': 'binomial', 'labels': 'multinomial'}

        super(MNIST, self).__init__(data, distributions=distributions,
                                    name=name, mode=mode, image_shape=(28, 28),
                                    **kwargs)

        if binarize:
            self.data[name] = _rng.binomial(
                p=self.data[name],
                size=self.data[name].shape, n=1).astype('float32')

    def get_data(self, source, mode):
        '''Fetch data from gzip pickle.

        Args:
            source (str): path to source.
            mode (str): `train`, `test`, or `valid`.

        '''
        with gzip.open(source, 'rb') as f:
            x = cPickle.load(f)

        if mode == 'train':
            X = np.float32(x[0][0])
            Y = np.float32(x[0][1])
        elif mode == 'valid':
            X = np.float32(x[1][0])
            Y = np.float32(x[1][1])
        elif mode == 'test':
            X = np.float32(x[2][0])
            Y = np.float32(x[2][1])
        else:
            raise ValueError()

        return X, Y


_classes = {'MNIST': MNIST}