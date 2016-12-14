'''
Street view house numbers truncated dataset
'''

from os import path
from scipy import io

from . import cifar
from ...utils import floatX


class SVHN(cifar.CIFAR):
    def __init__(self, name='SVHN', **kwargs):
        super(SVHN, self).__init__(name=name, **kwargs)

    def get_data(self, source, mode, greyscale=False):
        if mode == 'train':
            source_file = path.join(source, 'train_32x32.mat')
            data_dict = io.loadmat(source_file)
            X = data_dict['X']
            Y = data_dict['y']
        elif mode == 'valid':
            source_file = path.join(source, 'test_32x32.mat')
            data_dict = io.loadmat(source_file)
            X = data_dict['X']
            Y = data_dict['y']
        elif mode == 'test':
            source_file = path.join(source, 'test_32x32.mat')
            data_dict = io.loadmat(source_file)
            X = data_dict['X']
            Y = data_dict['y']
        else:
            raise ValueError()
        
        X = X.reshape((X.shape[0] * X.shape[1], X.shape[2], X.shape[3])).transpose(2, 1, 0)
        
        X_r = X[:, 0]
        X_g = X[:, 1]
        X_b = X[:, 2]
        
        if greyscale:
            X = 0.299 * X_r + 0.587 * X_b + 0.114 * X_g
        else:
            X = np.concatenate([X_r, X_b, X_g], axis=1)

        X = X.astype('float32') / float(255)
        X = (X - X.mean(0)) / X.std(0)

        return X, Y.astype('int64')

_classes = {'SVHN': SVHN}