'''
Generic dataset class
'''

from collections import OrderedDict
import numpy as np
import random


def resolve(c):
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

def make_one_hot(Y):
    class_list = np.unique(Y).tolist()
    n_classes = len(class_list)

    O = np.zeros((Y.shape[0], n_classes), dtype='float32')
    for idx in xrange(Y.shape[0]):
        i = class_list.index(Y[idx])
        O[idx, i] = 1.;
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

    Arguments:
        dataset: str
        train_batch_size (Optional) int.
        valid_batch_size (Optional) int.
        test_batch_size (Optional) int.

    Returns:
        train, valid, test Dataset objects.
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
    Arguments:
        idx: (Optional) list of list of int. Indices for train/valid/test
            datasets.

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
        N = dummy.n
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
    def __init__(self, batch_size=None, shuffle=True, inf=False, name='dataset',
                 mode=None, stop=None, balance=False, **kwargs):
        if batch_size is None:
            raise ValueError('Batch size argument must be given')

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.inf = inf
        self.name = name
        self.pos = 0
        self.stop = stop
        self.mode = mode
        self.balance = balance

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

class BasicDataset(Dataset):
    '''
    Dataset with numpy arrays as inputs. No visualization available.

    Arrays must be a dictionary of name/numpy array key/value pairs.
    '''
    def __init__(self, data, distributions=None, labels='label', name=None,
                **kwargs):
        if not isinstance(data, dict):
            raise ValueError('array argument must be a dict.')
        if name is None:
            name = data.keys()[0]

        super(BasicDataset, self).__init__(name=name, **kwargs)
        self.data = data
        self.n = None

        self.dims = dict()
        if distributions is None:
            self.distributions = dict()
        else:
            self.distributions = distributions

        if labels is None: labels = []

        for k, v in self.data.iteritems():
            if k == labels and len(v.shape) == 1:
                v = make_one_hot(v)
            elif len(v.shape) == 1:
                v = v[:, None]
            if self.stop is not None:
                v = v[:self.stop]
            self.data[k] = v

            if self.n is None:
                self.n = v.shape[0]
            else:
                if v.shape[0] != self.n:
                    raise ValueError('All input arrays must have the same'
                                    'number of samples (shape[0]), '
                                    '(%d vs %d)' % (self.n, v.shape[0]))
            self.dims[k] = v.shape[1]
            if not k in self.distributions.keys():
                self.distributions[k] = 'binomial'

        self.label_nums = self.data[labels].sum(axis=0)
        self.label_props = self.label_nums / float(self.n)

        self.labels = labels
        if self.balance:
            self.balance_labels()

        self.X = self.data[self.name]
        if labels in self.data.keys():
            self.Y = self.data[labels]

    def balance_labels(self):
        label_nums = self.data[self.labels].sum(axis=0)
        max_num = int(max(label_nums))

        dup_idx = []
        for i, label in enumerate(self.data[self.labels].T):
            l_sum = label.sum()
            if l_sum == max_num:
                continue
            idx = np.where(label == 1)[0].tolist()

            dup_idx = [idx[j] for j in range(max_num - len(idx))]
            print 'Balancing label %d by duplicating %d samples' % (i, len(dup_idx))

        dup_idx = np.unique(dup_idx)

        if len(dup_idx) > 0:
            for k, v in self.data.iteritems():
                self.data[k] = np.concatenate([self.data[k], self.data[k][dup_idx]])

        self.n += len(dup_idx)

        self.label_nums = self.data[self.labels].sum(axis=0)
        self.label_props = self.label_nums / float(self.n)

    def randomize(self):
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        for k in self.data.keys():
            self.data[k] = self.data[k][rnd_idx]

    def next(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.pos == -1:
            self.reset()

            if not self.inf:
                raise StopIteration

        rval = OrderedDict()

        for k, v in self.data.iteritems():
            rval[k] = v[self.pos:self.pos+batch_size]

        self.pos += batch_size
        if self.pos + batch_size > self.n:
            self.pos = -1

        return rval
