'''MNIST dataset.

'''

from collections import OrderedDict
import cPickle
import gzip
import logging
import numpy as np
import random

from . import Dataset
from ...utils import concatenate, scan, _rng
from ...utils.tools import resolve_path


logger = logging.getLogger(__name__)


class MNIST(Dataset):
    '''MNIST dataset iterator.

    Attributes:
        image_shape (tuple): dimensions of original images.

    '''
    
    distributions = {'input': 'binomial', 'labels': 'multinomial'}

    def __init__(self, restrict_digits=None, binarize=False, name='mnist',
                 **kwargs):
        '''Init function for MNIST.

        Args:
            restrict_digits (Optional[list]): list of digits to restrict
                iterator to.
            binarize (Optional[bool]): binarize the data.
            mode (str): `train`, `test`, or `valid`.
            name (str): name of dataset.
            **kwargs: extra keyword arguments passed to `get_data`

        '''
        super(MNIST, self).__init__(
            name=name, restrict_digits=restrict_digits, binarize=binarize,
            **kwargs)

    def fetch_data(self, mode, source=None, binarize=False, restrict_digits=None):
        '''Fetch data from gzip pickle.

        Args:
            mode (str): `train`, `test`, or `valid`.
            source (str): path to source.
            restrict_digits (Optional[list]): list of digits to restrict
                iterator to.
            binarize (Optional[bool]): binarize the data.

        '''
        source = source or self.source
        
        with gzip.open(source, 'rb') as f:
            x = cPickle.load(f)
        if restrict_digits is not None:
            for m in xrange(3):
                x_, y_ = x[m]
                x_ = np.array(x__ for i, x__ in enumerate(x_)
                              if y_[i] in restrict_digits)
                y_ = np.array(y__ for i, y__ in enumerate(y_)
                              if y_[i] in restrict_digits)
                x[m] = (x_, y_)
                
        if binarize:
            self.data[name] = _rng.binomial(
                p=self.data[name],
                size=self.data[name].shape, n=1).astype('float32')
            
        mode_dict = dict(train=0, valid=1, test=2)
        try:
            m = mode_dict[mode]
        except KeyError:
            raise KeyError('Unknown mode `{}`'.format(mode))
        
        images = np.float32(x[m][0])
        labels = np.float32(x[m][1])
        
        total_data = np.concatenate([np.float32(x[m][0]) for m in xrange(3)])
        mean_image = total_data.mean(0)
        var_image = total_data.std(0)

        self.load_data(mode, images=images)
        self.load_labels(mode, labels=labels)
        self.load_extras(mean_image=mean_image)


_classes = {'MNIST': MNIST}
