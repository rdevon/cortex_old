'''
Generic dataset class
'''


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


def load_data(dataset=None,
              train_batch_size=None,
              valid_batch_size=None,
              test_batch_size=None,
              **dataset_args):

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
