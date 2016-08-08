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


def fetch_basic_data():
    '''Fetch the basic dataset for demos.

    '''
    url = 'http://mialab.mrn.org/data/cortex/basic.zip'
    out_dir = resolve_path('$data')
    download_data(url, out_dir)
    unzip(path.join(out_dir, 'basic.zip'), out_dir)
    os.remove(path.join(out_dir, 'basic.zip'))

def resolve(c):
    '''Resolves the dataset class from string.

    This method only includes basic datasets, such as MNIST. For neuroimaging
    datasets, see `cortex.datasets.neuroimaging`


    Args:
        c (str): string to resolve.

    Returns:
        Dataset: dataset object resolved.

    '''
    from .basic.mnist import MNIST
    from .basic.caltech import CALTECH
    from .basic.uci import UCI
    from .basic.cifar import CIFAR

    r_dict = {
        'mnist': MNIST,
        'cifar': CIFAR,
        'caltech': CALTECH,
        'uci': UCI
    }

    C = r_dict.get(c, None)
    if C is None:
        raise ValueError('Dataset %s not supported' %c)

    return C

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

def load_data(dataset=None,
              train_batch_size=None,
              valid_batch_size=None,
              test_batch_size=None,
              resolve_dataset=None,
              **dataset_args):
    '''Load dataset with a predefined split.

    For these datasets, train/valid/test split has already been made.
    For the batch sizes, if any are None, the corresponding dataset
    will also be None.

    Args:
        dataset (str): name of the dataset
        train_batch_size (Optional[int])
        valid_batch_size (Optional[int])
        test_batch_size (Optional[int])

    Returns:
        Dataset: train dataset
        Dataset: valid dataset
        Dataset: test dataset

    '''

    if resolve_dataset is None:
        resolve_dataset = resolve

    if isinstance(dataset, str):
        C = resolve_dataset(dataset)
    else:
        C = dataset

    if train_batch_size is not None:
        train = C(batch_size=train_batch_size,
                  mode='train',
                  inf=False,
                  **dataset_args)
    else:
        train = None
    if valid_batch_size is not None:
        valid = C(batch_size=valid_batch_size,
                  mode='valid',
                  inf=False,
                  **dataset_args)
    else:
        valid = None
    if test_batch_size is not None:
        test = C(batch_size=test_batch_size,
                 mode='test',
                 inf=False,
                 **dataset_args)
    else:
        test = None

    return train, valid, test

def load_data_split(C, idx=None, dataset=None, **dataset_args):
    '''Load dataset and split.

    Args:
        idx: (Optional[list]): Indices for train/valid/test datasets.

    Returns:
        Dataset: train dataset
        Dataset: valid dataset
        Dataset: test dataset
        list: Indices for if split is created.

    '''
    train, valid, test, idx = make_datasets(C, **dataset_args)
    return train, valid, test, idx

def build_datasets(resolve_dataset, dataset=None, split=[0.7, 0.2, 0.1],
                  idx=None, train_batch_size=10, valid_batch_size=10,
                  test_batch_size=10, **dataset_args):
    C = resolve_class(dataset, _classes, __name__)

    if not hasattr(C, 'factory'):
        train, valid, test, idx = make_datasets(
            C, split=split, idx=idx, train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size, test_batch_size=test_batch_size,
            **dataset_args)
    else:
        train, valid, test, idx =  C.factory(
            split=split, idx=idx,
            batch_sizes=[train_batch_size, valid_batch_size, test_batch_size],
            **dataset_args)

    return OrderedDict(train=train, valid=valid, test=test, idx=idx)

def make_datasets(C, split=[0.7, 0.2, 0.1], idx=None,
                  train_batch_size=None,
                  valid_batch_size=None,
                  test_batch_size=None,
                  **dataset_args):
    '''Constructs train/valid/test datasets with idx or split.

    If idx is None, use split ratios to create indices.

    Arguments:
        C (Dataset).
        split (Optional[list]: Split ratios over total.
        idx (Optional[list]: Indices for train/valid/test datasets.
        train_batch_size (Optional[int])
        valid_batch_size (Optional[int])
        test_batch_size (Optional[int])

    Returns:
        Dataset: train dataset
        Dataset: valid dataset
        Dataset: test dataset
        list: Indices for if split is created.

    '''
    if idx is None:
        assert split is not None
        if round(np.sum(split), 5) != 1. or len(split) != 3:
            raise ValueError(split)
        dummy = C(batch_size=1, **dataset_args)
        N = dummy.X.shape[0]
        idx = range(N)
        random.shuffle(idx)
        split_idx = []
        accum = 0
        for s in split:
            s_i = int(s * N + accum)
            split_idx.append(s_i)
            accum += s_i

        train_idx = idx[:split_idx[0]]
        valid_idx = idx[split_idx[0]:split_idx[1]]
        test_idx = idx[split_idx[1]:]
        idx = [train_idx, valid_idx, test_idx]
    if train_batch_size is not None and len(train_idx) > 0:
        train = C(idx=idx[0], batch_size=train_batch_size, mode='train',
                  **dataset_args)
    else:
        train = None
    if valid_batch_size is not None and len(valid_idx) > 0:
        valid = C(idx=idx[1], batch_size=valid_batch_size, mode='valid',
                  **dataset_args)
    else:
        valid = None
    if test_batch_size is not None and len(test_idx) > 0:
        test = C(idx=idx[2], batch_size=test_batch_size, mode='test',
                 **dataset_args)
    else:
        test = None

    return train, valid, test, idx


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
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(
                '.'.join([self.__module__, self.__class__.__name__]))
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

    def reset(self):
        '''Reset the dataset post-epoch.

        '''
        self.pos = 0
        if self.shuffle:
            self.randomize()

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
            for k, a in attributes.iteritems())
        attr_str = ''
        for k, a in attributes.iteritems():
            attr_str += '\n\t%s: %s' % (k, a)
        s = ('<Dataset %s: %s>' % (self.__class__.__name__, attr_str))
        return s

    def get_dim(self, key):
        dim_map = {
            'input': self.dims[self.name]
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
                balance=False, one_hot=True, transpose=None, **kwargs):
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
        if name is None:
            name = data.keys()[0]

        super(BasicDataset, self).__init__(name=name, **kwargs)
        self.data = data
        self.n_samples = None
        self.balance = balance
        self.transpose = transpose

        if distributions is not None:
            self.distributions.update(**distributions)

        if labels not in self.data.keys():
            labels = None

        for k, v in self.data.iteritems():
            if k == labels and one_hot and len(v.shape) == 1:
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
        self.mean_image = self.X.mean(axis=0)
        self.labels = labels

        if self.labels is not None:
            self.label_nums = self.data[labels].sum(axis=0)
            self.label_props = self.label_nums / float(self.n_samples)

            if self.balance:
                self.balance_labels()

            if self.labels in self.data.keys():
                self.Y = self.data[labels]

        if self.shuffle:
            self.randomize()

        self.register()

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
            for k, v in self.data.iteritems():
                self.data[k] = np.concatenate([self.data[k], self.data[k][dup_idx]])

        self.n_samples += len(dup_idx)

        self.label_nums = self.data[self.labels].sum(axis=0)
        self.label_props = self.label_nums / float(self.n_samples)

    def randomize(self):
        '''Randomizes the dataset

        '''
        rnd_idx = np.random.permutation(np.arange(0, self.n_samples, 1))
        for k in self.data.keys():
            self.data[k] = self.data[k][rnd_idx]

    def next(self, batch_size):
        '''Draws the next batch of data samples.

        Arguments:
            batch_size (int).

        Returns:
            dict: Dictionary of data.

        '''

        if self.pos == -1:
            self.reset()
            raise StopIteration

        rval = OrderedDict()

        for k, v in self.data.iteritems():
            v = v[self.pos:self.pos+batch_size]
            if self.transpose is not None and k in self.transpose.keys():
                v = v.transpose(self.transpose[k])
            rval[k] = v

        self.pos += batch_size
        if self.pos + batch_size > self.n_samples:
            self.pos = -1

        return rval

    def __str__(self):
        attributes = self.__dict__
        attributes = dict(
            (k, '<numpy.ndarray: {shape: %s}>' % (a.shape,)) if isinstance(a, np.ndarray)
            else (k, a)
            for k, a in attributes.iteritems())
        attributes['data'] = dict(
            (k, '<numpy.ndarray: {shape: %s}>' % (a.shape,))
            for k, a in attributes['data'].iteritems())
        attr_str = ''
        for k, a in attributes.iteritems():
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
        d = self.next(10)
        tensors = OrderedDict()
        for k, v in d.iteritems():
            self.logger.debug('Data mode `%s` has shape %s. '
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
                raise ValueError('dtype %s not supported' % v.dtype)

            X = C(self.name + '.' + k, dtype=dtype)
            tensors[k] = X
        self.logger.debug('Dataset has the following tensors: %s with types %s'
                          % (tensors, [inp.dtype for inp in tensors.values()]))
        self.reset()
        return tensors


_classes = {'BasicDataset': BasicDataset}