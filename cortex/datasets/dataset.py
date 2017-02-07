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
        Y (numpy.array): N x 1 array of integers.

    Returns:
        numpy.array: N x n_labels array of ones and zeros.

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
        raise TypeError('`make_one_hot` supports 1 or 2 dims, (%d)' % Y.ndim)
    else:
        reshape = None

    O = np.zeros((Y.shape[0], n_classes), dtype='float32')
    for idx in xrange(Y.shape[0]):
        try:
            i = class_list.index(Y[idx])
        except ValueError:
            raise ValueError('Class list is missing elements')
        O[idx, i] = 1.;

    if reshape is not None:
        O = O.reshape(reshape + (n_classes,))
    return O


class Dataset(object):
    '''Base dataset iterator class.

    Attributes:
        batch_size (int): batch size for the iterator.
        shuffle (bool): shuffle the dataset after each epoch.
        inf (bool): reset the dataset after iterating through who set.
        name (str): name of the dataset.
        pos (int): current position of the iterator.
        stop (int): stop the dataset at this index when loading.
        mode (str): usually train, test, valid.
        dims (dict): dictionary of data dimensions.
        distributions (dict): dictionary of strings. See `models.distributions`
            for details.

    '''
    def __init__(self, shuffle=True, inf=False, name='dataset',
                 mode=None, stop=None, **kwargs):
        '''Init function for Dataset

        Args:
            shuffle (bool): shuffle the dataset after each epoch.
            inf (bool): reset the dataset after iterating through who set.
            name (str): name of the dataset.
            pos (int): current position of the iterator.
            stop (int): stop the dataset at this index when loading.
            mode (str): usually train, test, valid.
            **kwargs: keyword arguments not used

        Returns:
            dict: leftover keyword arguments.

        '''
        if not hasattr(self, 'logger'): self.logger = get_class_logger(self)
        self.logger.debug('Forming dataset %r with name %s' % (
            self.__class__, name))

        self.shuffle = shuffle
        self.inf = inf
        self.name = name
        self.pos = 0
        self.stop = stop
        self.mode = mode
        self.dims = dict()
        self.distributions = dict()

        return kwargs

    def randomize(self):
        '''Randomize the dataset.

        '''
        return

    def reset(self, batch_size=None):
        '''Reset the dataset post-epoch.

        '''
        self.pos = 0
        if self.shuffle: self.randomize()

    def __iter__(self):
        '''Iterator.

        '''
        return self

    def save_images(self, *args):
        '''Save images.

        '''
        pass

    def __str__(self):
        attributes = self.__dict__
        attributes = dict(
            (k, '<numpy.ndarray: {shape: %s}>' % (a.shape,)) if isinstance(a, np.ndarray)
            else (k, a)
            for k, a in attributes.items())
        attr_str = ''
        for k, a in attributes.items():
            attr_str += '\n\t%s: %s' % (k, a)
        s = ('<Dataset %s: %s>' % (self.__class__.__name__, attr_str))
        return s

    def get_dim(self, key):
        dim_map = {
            'input': self.dims[self.name],
            'input_centered': self.dims[self.name]
        }
        dim_map.update(**self.dims)

        return dim_map[key]


class BasicDataset(Dataset):
    '''
    Dataset with numpy arrays as inputs. No visualization available.

    Arrays must be a dictionary of name/numpy array key/value pairs.

    Attributes:
        data (dict): dictionary of numpy.array.
        n_samples (int): number of data samples.
        X (numpy.array): primary data.
        Y (Optional[numpy.array]): If not None, lables.
        mean_image (numpy.array): mean image of primary data.
        balance (bool): replicate samples to balance the dataset.

    '''
    _has_split = False
    _viz = []

    def __init__(self, data, distributions=None, labels='labels', name=None,
                balance=False, one_hot=True, transpose=None, check_data=False,
                process_centered=True, idx_ref=None, **kwargs):
        '''Init function for BasicDataset.

        Args:
            data (dict): Dictionary of np.array. Keys are data name, value is
                the actual data.
            distributions (dict): See `models.distributions` for more details.
            labels (str): key for the labels.
            name: (Optional[str]): Name of the dataset. Should be one of the
                keys in data.
            balance (bool): replicate samples to balance the dataset.
            one_hot (bool): convert labels to one-hot.
            **kwargs: extra arguments to pass to Dataset constructor.

        '''
        if not isinstance(data, dict):
            raise ValueError('array argument must be a dict.')
        if name is None: name = data.keys()[0]

        super(BasicDataset, self).__init__(name=name, **kwargs)
        self.data = data
        self.n_samples = None
        self.balance = balance
        self.transpose = transpose

        if distributions is not None:
            self.distributions.update(**distributions)

        if labels not in self.data.keys():
            labels = None

        for k, v in self.data.items():
            if k == labels and one_hot and len(v.shape) == 1:
                v = make_one_hot(v)
            elif distributions[k] == 'multinomial':
                v = make_one_hot(v)
            elif len(v.shape) == 1:
                v = v[:, None]
            if self.stop is not None:
                v = v[:self.stop]
            self.data[k] = v

            if self.n_samples is None:
                self.n_samples = v.shape[0]
            else:
                if v.shape[0] != self.n_samples:
                    raise ValueError('All input arrays must have the same'
                                    'number of samples (shape[0]), '
                                    '(%d vs %d)' % (self.n_samples, v.shape[0]))
            self.dims[k] = v.shape[-1]
            if not k in self.distributions.keys():
                self.distributions[k] = 'binomial'

        self.X = self.data['input']
        if not hasattr(self, 'mean_image'):
            self.mean_image = self.X.mean(axis=0)
        self.var_image = self.X.std(axis=0)
        self.variance = self.X.std()
        self.labels = labels
        if process_centered:
            self.data['input_centered'] = self.data['input'] - self.mean_image
            self.dims['input_centered'] = self.dims['input']
            self.distributions['input_centered'] = self.distributions['input']

        if self.labels is not None:
            self.label_nums = self.data[labels].sum(axis=0)
            self.label_props = self.label_nums / float(self.n_samples)

            if self.balance:
                self.balance_labels()

            if self.labels in self.data.keys():
                self.Y = self.data[labels]

        self.finish_setup()

        if check_data: self.check()
        
        self.idx_ref = idx_ref
        if self.idx_ref is None:
            if not hasattr(self, 'idx'): self.idx = range(self.n_samples)
        else:
            self.idx = self.manager.datasets[self.idx_ref][self.mode].idx
        if self.shuffle: self.randomize()

        self.register()

    def finish_setup(self):
        return
    
    def check(self):
        self.logger.info('Checking data for {}'.format(self.name))
        for k in self.data.keys():
            self.logger.info('Checking data `{}`'.format(k))
            data = self.data[k]
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

    def register(self):
        from .. import _manager as manager

        datasets = manager.datasets
        if self.name in datasets.keys():
            if self.mode in datasets[self.name].keys():
                self.logger.warn(
                    'Dataset with name `%s` and mode `%s` already found: '
                    'overwriting. Use `cortex.manager.remove_dataset` to avoid '
                    'this warning' % (self.name, self.mode))
            datasets[self.name][self.mode] = self
        else:
            d = {('%s' % self.mode):self}
            datasets[self.name] = d
            datasets[self.name]['dims'] = self.dims
            datasets[self.name]['distributions'] = self.distributions
            datasets[self.name]['tensors'] = self.make_tensors()

    def copy(self):
        return copy.deepcopy(self)

    def balance_labels(self):
        '''Balance the dataset.

        '''
        self.logger.debug('Balancing dataset %s' % self.name)
        label_nums = self.data[self.labels].sum(axis=0)
        max_num = int(max(label_nums))

        dup_idx = []
        for i, label in enumerate(self.data[self.labels].T):
            l_sum = label.sum()
            if l_sum == max_num:
                continue
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

    def randomize(self):
        '''Randomizes the dataset

        '''
        if self.idx_ref is None:
            self.idx = np.random.permutation(
                np.arange(0, self.n_samples, 1)).tolist()
        
    def next(self, batch_size):
        '''Draws the next batch of data samples.

        Arguments:
            batch_size (int).

        Returns:
            dict: Dictionary of data.

        '''

        if self.pos is None:
            self.reset()
            raise StopIteration

        rval = OrderedDict()

        idx = [self.idx[p] for p in range(self.pos, self.pos + batch_size)]
        for k, v in self.data.items():
            v = v[idx]
            if self.transpose is not None and k in self.transpose.keys():
                v = v.transpose(self.transpose[k])
            rval[k] = v

        self.pos += batch_size
        if self.pos + batch_size > self.n_samples:
            self.pos = None

        return rval

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

    def set_link_value(self, key):
        k_, name = key.split('.')
        if name in self.data.keys():
            if k_ == 'dim':
                return self.dims[name]
            elif k_ == 'distribution':
                return self.distributions[name]
            else:
                raise KeyError
        else:
            raise KeyError

    def make_tensors(self):
        '''Forms the tensors from the dataset

        '''
        self.logger.debug('Forming tensors for dataset %s' % self.name)
        try:
            d = self.next(10)
        except StopIteration:
            d = self.next(10)
        tensors = OrderedDict()
        for k, v in d.items():
            self.logger.info('Data mode `%s` has shape %s. '
                             '(tested with batch_size 10)' % (k, v.shape))
            if v.ndim == 1:
                C = T.vector
            elif v.ndim == 2:
                C = T.matrix
            elif v.ndim == 3:
                C = T.tensor3
            else:
                raise ValueError('Data dim over 3 not supported.')

            if v.dtype == floatX:
                dtype = floatX
            elif v.dtype == intX:
                dtype = intX
            else:
                raise ValueError('dtype %s not supported (%s)' % (v.dtype, k))

            X = C(self.name + '.' + k, dtype=dtype)
            tensors[k] = X
        self.logger.debug(
            'Dataset has the following tensors: %s with types %s'
            % (tensors, [inp.dtype for inp in tensors.values()]))
        self.reset()
        return tensors


_classes = {'BasicDataset': BasicDataset}
