'''
SNP dataset class.
'''

from collections import OrderedDict
import numpy as np
from scipy.io import loadmat
from . import BasicDataset, Dataset
from utils.tools import (
    warn_kwargs
)


class SNP(BasicDataset):
    '''SNP dataset class.

    Currently only handled continuous preprocessed data.
    Discrete data TODO
    '''
    def __init__(self, source=None, name='snp', mode='train', convert_one_hot=True, idx=None, **kwargs):
        '''Initialize the SNP dataset.

        Arguments:
            source: str. Path to source file.
            name: str. ID of dataset.
            idx: (Optional) list. List of indices for train/test/validation
                split.
        '''
        if source is None:
            raise ValueError('No source provided')

        # Fetch SNP data from "source"
        X, Y = self.get_data(source)
        data = {name: X, 'label': Y}

        # balance data for traning, valid, and test parts
        balance = False
        if idx is not None:
            balance=True
            data[name] = data[name][idx]
            data['label'] = data['label'][idx]

        distributions = {name: 'gaussian', 'label': 'multinomial'}
        super(SNP, self).__init__(data, name=name, balance=balance, distributions=distributions,  mode=mode, **kwargs)

        self.mean_image = self.data[name].mean(axis=0)

    def get_data(self, source):
        '''Fetch the data from source.

        Genetic data is in the matrix format with size Subjec*SNP
        SNP can be either preprocessed or notprocessed
        Labels is a vector with diagnosis info
        Patients are coded with 1 and health control coded with 2

        Arguments:
           source: dict. file names of genetic data and labels
                  {'snp' key for genetic data
                    'labels' key for diagnosis }
        '''
        from utils.tools import get_paths
        data_path = get_paths()['$snp_data']
        print('Loading genetic data from %s' % data_path)
        X = loadmat(data_path + '/' + source['snp'])
        Y = loadmat(data_path + '/' + source['label'])
        X = np.float32(X[X.keys()[2]])
        Y = np.float32(Y[Y.keys()[0]])
        Y.resize(max(Y.shape,))
        return X, Y

    def reset(self):
        '''Reset the iterator'''
        self.pos = 0
        if self.shuffle:
            self.randomize()
