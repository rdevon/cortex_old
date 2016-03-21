'''
Generic dataset class
'''

from collections import OrderedDict


def load_data(dataset=None,
              train_batch_size=None,
              valid_batch_size=None,
              test_batch_size=None,
              **dataset_args):

    from caltech import CALTECH
    from cifar import CIFAR
    from mnist import MNIST
    from uci import UCI
    from snp import SNP

    if dataset == 'mnist':
        C = MNIST
    elif dataset == 'cifar':
        C = CIFAR
    elif dataset == 'caltech':
        C = CALTECH
    elif dataset == 'uci':
        C = UCI
    elif dataset == 'snp':
        C = SNP


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
