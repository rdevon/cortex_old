'''
Generic dataset class
'''

from collections import OrderedDict
import copy
import logging
import numpy as np
import os
from os import path
import random
from theano import tensor as T

from ..base import Base
from .. import namespaces
from ..utils import floatX, intX
from ..utils.tools import resolve_path
from ..utils.extra import download_data, unzip
from ..utils.logger import get_class_logger


def fetch_basic_data():
    '''Fetch the basic dataset for demos.

    '''
    url = 'http://mialab.mrn.org/data/cortex/basic.zip'
    out_dir = resolve_path('$data')
    download_data(url, out_dir)
    unzip(path.join(out_dir, 'basic.zip'), out_dir)
    os.remove(path.join(out_dir, 'basic.zip'))

def make_one_hot(Y, n_classes=None):
    '''Makes integer label data into one-hot.

    Args:
        Y (numpy.ndarray): N x 1 array of integers.
        n_classes (Optional[int]): Number of classes.

    Returns:
        numpy.ndarray: N x n_labels array of ones and zeros.

    '''
    if n_classes is None:
        class_list = np.unique(Y).tolist()
        n_classes = len(class_list)
    else:
        class_list = range(n_classes)

    if Y.ndim == 2:
        reshape = Y.shape
        Y = Y.reshape((Y.shape[0] * Y.shape[1]))
    elif Y.ndim > 2:
        raise TypeError('`make_one_hot` supports 1 or 2 dims, ({})'.format(
            Y.ndim))
    else:
        reshape = None

    O = np.zeros((Y.shape[0], n_classes), dtype='float32')
    for idx in xrange(Y.shape[0]):
        try:
            i = class_list.index(Y[idx])
        except ValueError:
            raise ValueError('Class list is missing elements')
        O[idx, i] = 1.;

    if reshape is not None: O = O.reshape(reshape + (n_classes,))
    return O


class DataIterator(object):
    '''Dataset iterator class.
    
    Iterates through the data.

    Attributes:
        pos (int): current position of the iterator.
        mode (str): usually train, test, valid.

    '''
    def __init__(self, mode=None, stop=None, shuffle=False):
        '''Init function for Dataset

        Args:
            inf (bool): reset the dataset after iterating through who set.
            mode (str): usually train, test, valid.

        '''

        self.pos = 0
        self.mode = mode
        self.n_samples = None
        self.stop = stop
        self.shuffle = shuffle
        self._data = {}
        self.idx = None
                
    def set_args(self, **kwargs):
        for k, v in kwargs.items():
            if k in ['stop', 'shuffle']:
                setattr(self, k, v)
            else:
                msg = '`{0}` object has no attribute `{1}`'
                raise AttributeError(msg.format(type(self).__name__, k))

    def reset(self):
        '''Reset the dataset post-epoch.

        '''
        self.pos = 0
        if self.shuffle: self.randomize()
        
    def hard_reset(self):
        self.pos = 0
        self._data = {}
        
    def randomize(self):
        '''Randomizes the dataset

        '''
        n_samples = self.stop or self.n_samples
        self.idx = np.random.permutation(np.arange(0, n_samples, 1))
        
    def add(self, ds_name, k, v):
        name = '{0}_{1}'.format(ds_name, k)
        if name in self._data.keys():
            raise ValueError('`{0}` already in mode `{1}`'.format(
                name, self.mode))
        self._data[name] = v

    def next(self, batch_size):
        '''Draws the next batch of data samples.

        Arguments:
            batch_size (int).

        Returns:
            dict: Dictionary of data.

        '''
        self.idx = self.idx or range(self.n_samples)
        n_samples = len(self.idx)

        if self.pos == -1:
            self.reset()
            raise StopIteration

        rval = OrderedDict()

        for k, v in self._data.items():
            v = v[self.idx][self.pos:self.pos+batch_size]
            rval[k] = v

        self.pos += batch_size
        if self.pos + batch_size > n_samples: self.pos = -1

        return rval

    def __iter__(self):
        '''Iterator.

        '''
        return self

    def __str__(self):
        attributes = self.__dict__
        attributes = dict(
            (k, '<numpy.ndarray: \{shape: {}\}>'.format(a.shape))
            if isinstance(a, np.ndarray)
            else (k, a)
            for k, a in attributes.items())
        attr_str = ''
        for k, a in attributes.items():
            attr_str += '\n\t{0}: {1}'.format(k, a)
        s = ('<Dataset {}: {}>'.format(self.__class__.__name__, attr_str))
        return s
    

class Dataset(Base):
    '''
    Dataset with numpy arrays as inputs. No visualization available.

    Arrays must be a dictionary of name/numpy array key/value pairs.

    Attributes:
        batch_size (int): batch size for the iterator.
        n_samples (int): number of data samples.
        X (numpy.array): primary data.
        Y (Optional[numpy.array]): If not None, lables.
        mean_image (numpy.array): mean image of primary data.
        balance (bool): replicate samples to balance the dataset.
        dims (dict): dictionary of data dimensions.
        distributions (dict): dictionary of strings. See `models.distributions`
            for details.

    '''
    
    distributions = None

    def __init__(self, source=None, modes=None, labels='labels',
                 name=None, one_hot=True, **kwargs):
        '''Init function for BasicDataset.

        Args:
            source (str): Path to source file or directory.
            modes (Optional[list]): List of modes to load.
            name: (Optional[str]): Name of the dataset. Should be one of the
                keys in data.
            balance (bool): replicate samples to balance the dataset.
            one_hot (bool): convert labels to one-hot.
            **kwargs: extra arguments to pass to `get_data`.

        '''
        modes = modes or ['train']
        if not isinstance(modes, list): modes = [modes]
        super(Dataset, self).__init__(name=name)
        self.logger.debug('Forming dataset {0} with name {1}'.format(
            self.__class__, name))
        
        if self.distributions is None:
            raise ValueError('Dataset class {} has unset distributions'.format(
                self.__class__))
        if source is None: raise TypeError('No source file provided')
        
        self.source = resolve_path(source)
        self.dims = {}
        self.shapes = {}
        self.dtypes = {}
        self._data = {}
        self._n_samples = {}

        self.logger.info('Loading {name} ({modes}) from {source} with '
                         'arguments {kwargs}'.format(
            name=name, modes=modes, source=self.source, kwargs=kwargs))
        
        for mode in modes:
            self.fetch_data(mode, **kwargs)
            
        self.make_tensors()

    def todo(self):
        self.balance = balance

        if not hasattr(self, 'mean_image'): self.mean_image = self.X.mean(axis=0)
        self.var_image = self.X.std(axis=0)
        self.variance = self.X.std()

        if self.labels is not None:
            self.label_nums = self.data[labels].sum(axis=0)
            self.label_props = self.label_nums / float(self.n_samples)
            if self.balance: self.balance_labels()
            if self.labels in self.data.keys(): self.Y = self.data[labels]

        if check_data: self.check()
        
    def load_extras(self, **kwargs):
        pass
    
    def load_data(self, mode, **kwargs):
        if mode not in self.manager.data.keys():
            self.manager.add_data_mode(mode)
            
        for k, v in kwargs.items():
            if self._n_samples.get(mode, None) is None:
                self._n_samples[mode] = v.shape[0]
            elif v.shape[0] != self._n_samples[mode]:
                raise ValueError('All input arrays must have the same'
                                 'number of samples (shape[0]), '
                                 '({0} vs {1})'.format(self._n_samples[mode],
                                                     v.shape[0]))
            if k in self._data.keys() and mode in self._data[k]:
                raise ValueError('Data `{0}` already exists for mode `{1}`. '
                                 'Cannot overwrite'.format(k, mode))
            elif k not in self._data.keys():
                self._data[k] = {}
            if self.dims.get(k, None) is not None:
                if self.dims[k] != v.shape[-1]:
                    raise ValueError('Dimensions of `{0}` does not match '
                                     '({1} vs {2})'.format(k, v.shape[-1],
                                                           self.dims[k]))
            else:
                self.dims[k] = v.shape[-1]
            if self.shapes.get(k, None) is not None:
                if self.shapes[k] != v.shape[1:]:
                    raise ValueError('Shapes of `{0}` does not match '
                                     '({1} vs {2})'.format(k, v.shape[1:],
                                                           self.shapes[k]))
            else:
                self.shapes[k] = v.shape[1:]
            if self.dtypes.get(k, None) is not None:
                if self.dtypes[k] != v.dtype:
                    raise ValueError('Data type of `{0}` does not match '
                                     '({1} vs {2})'.format(k, v.dtyle,
                                                           self.dtyles[k]))
            else:
                self.dtypes[k] = v.dtype
            self._data[k][mode] = v
            self.manager.data[mode].add(self.name, k, v)
            
    def load_labels(self, mode, one_hot=True, **kwargs):
        for k, v in kwargs.items():
            if one_hot: kwargs[k] = make_one_hot(v)
        self.load_data(mode, **kwargs)

    def finish_setup(self):
        return

    def copy(self):
        return copy.deepcopy(self)

    def balance(self):
        '''Balance the dataset.

        '''
        self.logger.debug('Balancing dataset %s' % self.name)
        label_nums = self.data[self.labels].sum(axis=0)
        max_num = int(max(label_nums))

        dup_idx = []
        for i, label in enumerate(self.data[self.labels].T):
            l_sum = label.sum()
            if l_sum == max_num: continue
            idx = np.where(label == 1)[0].tolist()

            dup_idx = [idx[j] for j in range(max_num - len(idx))]
            self.logger.debug('Balancing label %d by duplicating %d samples'
                             % (i, len(dup_idx)))

        dup_idx = np.unique(dup_idx)

        if len(dup_idx) > 0:
            for k, v in self.data.items():
                self.data[k] = np.concatenate([self.data[k], self.data[k][dup_idx]])

        self.n_samples += len(dup_idx)
        self.label_nums = self.data[self.labels].sum(axis=0)
        self.label_props = self.label_nums / float(self.n_samples)
        
    def check(self):
        self.logger.info('Checking data for {}'.format(self.name))
        for k in self.data.keys():
            self.logger.info('Checking data `{}`'.format(k))
            data = self._data[k]
            dist = self.distributions[k]
            dim = self.dims[k]
            
            mi = data.min()
            ma = data.max()
            mean = data.mean()
            std = data.std()
            
            hasnan = np.any(np.isnan(data))
            hasinf = np.any(np.isinf(data))
            self.logger.info('Data stats for `{0}`: dist: {1}, dim: {2}, '
                             'min: {3:.2e}, max: {4:.2e}, mean: {5:.2e}, '
                             'std: {6:.2e}, has nans: {7}, has infs: {8}'.format(
                                k, dist, dim, mi, ma, mean, std, hasnan, hasinf))
        self.logger.info('Done checking data.')

    def __str__(self):
        attributes = self.__dict__
        attributes = dict(
            (k, '<numpy.ndarray: {shape: %s}>' % (a.shape,))
            if isinstance(a, np.ndarray)
            else (k, a)
            for k, a in attributes.items())
        attributes['data'] = dict(
            (k, '<numpy.ndarray: {shape: %s}>' % (a.shape,))
            for k, a in attributes['data'].items())
        attr_str = ''
        for k, a in attributes.items():
            attr_str += '\n\t%s: %s' % (k, a)
        s = ('<Dataset %s: %s>' % (self.__class__.__name__, attr_str))
        return s

    def make_tensors(self):
        '''Forms the tensors from the dataset

        '''
        self.logger.debug('Forming tensors for dataset %s' % self.name)
        
        tensors = OrderedDict()
        
        dim_dict = {1: T.vector, 2: T.matrix, 3: T.tensor3}
        
        for k, shape in self.shapes.items():
            self.logger.info('Data mode `{0}` has shape {1}.'.format(k, shape))
            try:
                C = dim_dict[len(shape)+1]
            except KeyError:
                raise ValueError(
                    'Data dim over 3 not supported (got {}).'.format(
                        len(shape)+1))

            dtype = self.dtypes[k]
            if dtype not in [floatX, intX]:
                raise ValueError('dtype {} not supported'.format(dtype))

            X = C(self.name + '.' + k, dtype=dtype)
            tensors[k] = X
            
            if self.name not in self.manager.tensors:
                self.manager.tensors[self.name] = namespaces.TensorNamespace()
            if k not in self.manager.tensors[self.name]:
                self.manager.tensors[self.name][k] = X
        self.logger.debug('Dataset has the following tensors: {0} with types '
                          '{1}'.format(tensors,
                                       [inp.dtype for inp in tensors.values()]))
        

_classes = {'Dataset': Dataset}
