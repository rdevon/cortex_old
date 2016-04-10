'''
Generic dataset class
'''

from collections import OrderedDict
import numpy as np
import random


def load_data(dataset=None,
              train_batch_size=None,
              valid_batch_size=None,
              test_batch_size=None,
              **dataset_args):
    '''Load dataset with a predefined split.

    For these datasets, train/valid/test split has already been made.
    For the batch sizes, if any are None, the corresponding dataset
    will also be None.

    Arguments:
        dataset: str
        train_batch_size (Optional) int.
        valid_batch_size (Optional) int.
        test_batch_size (Optional) int.

    Returns:
        train, valid, test Dataset objects.

    '''

    from caltech import CALTECH
    from cifar import CIFAR
    from mnist import MNIST
    from uci import UCI

    if dataset == 'mnist':
        C = MNIST
    elif dataset == 'cifar':
        C = CIFAR
    elif dataset == 'caltech':
        C = CALTECH
    elif dataset == 'uci':
        C = UCI

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

    Arguments:
        idx: (Optional) list of list of int. Indices for train/valid/test
            datasets.
        C: Dataset Object.

    Returns:
        train, valid, test Dataset objects.
        idx: Indices for if split is created.
    '''

    train, valid, test, idx = make_datasets(C, **dataset_args)
    return train, valid, test, idx

def make_datasets(C, split=[0.7, 0.2, 0.1], idx=None,
                  train_batch_size=None,
                  valid_batch_size=None,
                  test_batch_size=None,
                  **dataset_args):
    '''Constructs train/valid/test datasets with idx or split.

    If idx is None, use split ratios to create indices.

    Arguments:
        C: Dataset class.
        split: (Optional) list of float. Split ratios over total.
        idx: (Optional) list of list of int. Indices for train/valid/test
            datasets.
        train_batch_size (Optional) int.
        valid_batch_size (Optional) int.
        test_batch_size (Optional) int.

    Returns:
        train, valid, test Dataset objects.
        idx: Indices for if split is created.
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
        train = C(idx=idx[0], batch_size=train_batch_size, **dataset_args)
    else:
        train = None
    if valid_batch_size is not None and len(valid_idx) > 0:
        valid = C(idx=idx[1], batch_size=valid_batch_size, **dataset_args)
    else:
        valid = None
    if test_batch_size is not None and len(test_idx) > 0:
        test = C(idx=idx[2], batch_size=test_batch_size, **dataset_args)
    else:
        test = None

    return train, valid, test, idx


class Dataset(object):
    def __init__(self, batch_size=None, shuffle=True, inf=False, name='dataset',
                 stop=None, **kwargs):
        if batch_size is None:
            raise ValueError('Batch size argument must be given')

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.inf = inf
        self.name = name
        self.pos = 0
        self.stop = stop

        return kwargs

    def randomize(self):
        return

    def reset(self):
        self.pos = 0
        if self.shuffle:
            self.randomize()

    def __iter__(self):
        return self

    def save_images(self, *args):
        pass

class BasicDataset(object):
    '''
    Dataset with numpy arrays as inputs. No visualization available.

    Arrays must be a dictionary of name/numpy array key/value pairs.
    '''
    def __init__(self, arrays, distributions=None, name=None, **kwargs):
        if not isinstance(arrays, dict):
            raise ValueError('array argument must be a dict.')
        if name is None:
            name = arrays.keys()[0]

        super(BasicDataset, self).__init__(name=name, **kwargs)
        self.arrays = arrays
        self.n = None

        self.dims = dict()
        if self.distributions is None:
            self.distributions = dict()
        else:
            self.distributions = distributions

        for a_name, array in self.arrays.iteritems():
            if self.n is None:
                self.n = array.shape[0]
            else:
                if array.shape[0] != self.n:
                    raise ValueError('All input arrays must have the same'
                                    'number of samples (shape[0]), '
                                    '(%d vs %d)' % (self.n, array.shape[0]))
            self.dims[a_name] = array.shape[1]
            if not a_name in self.distributions.keys():
                self.distributions[a_name] = 'binomial'

    def randomize(self):
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        for a_name in self.arrays.keys():
            self.arrays[a_name] = self.arrays[a_name][rnd_idx, :]

    def next(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.pos == -1:
            self.reset()

            if not self.inf:
                raise StopIteration

        rval = OrderedDict()

        for a_name, array in self.arrays.iteritems():
            rval[a_name] = array[self.pos:self.pos+batch_size]

        self.pos += batch_size
        if self.pos + batch_size > self.n:
            self.pos = -1

        return rval
